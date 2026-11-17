"""Build the submission report from checked, immutable numerical evidence.

Run from any directory with the optional report_tools/requirements.txt environment.
This script does not load a model, fit parameters, score shipments or alter the
frozen application. Embedded local fonts and invariant metadata stabilize layout.
"""
from pathlib import Path
from xml.sax.saxutils import escape
import argparse
import csv
import hashlib
import json
from importlib.metadata import version

from reportlab.lib import colors
from reportlab.lib.enums import TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate, Paragraph, Table, TableStyle, Spacer, PageBreak

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'reports/evidence'
FONT_ROOT = Path(__file__).resolve().parent / 'fonts'
W, H = A4
MARGIN = 48.0
CONTENT_WIDTH = W - 2 * MARGIN
TOP = H - 60.0
BOTTOM = 55.0
BLACK = colors.black


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def build(output):
    if version('reportlab') != '4.4.9':
        raise RuntimeError('Use the pinned report_tools/requirements.txt environment.')
    manifest = read_json(EVIDENCE / 'evidence_manifest.json')
    for item in manifest['files']:
        path = EVIDENCE / item['file']
        if hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError(f'Report evidence checksum mismatch: {item["file"]}')
    summary = read_json(EVIDENCE / 'holdout_summary.json')
    selection = read_json(EVIDENCE / 'validation_summary.json')
    plan = read_json(EVIDENCE / 'split_manifest.json')
    audit = read_json(EVIDENCE / 'dataset_audit.json')
    cfg = read_json(EVIDENCE / 'simulation_config.json')
    review = read_json(EVIDENCE / 'development_review.json')
    example = read_json(EVIDENCE / 'first_holdout_explanation.json')
    verification = read_json(ROOT / 'reports/verification/audit_summary.json')
    metrics = {r['predictor']: r for r in summary['metrics']}
    baseline_validation = {r['baseline']: r for r in read_csv(EVIDENCE / 'validation_baselines.csv') if r['scope'] == 'pooled_validation'}
    case = next(r for r in read_csv(EVIDENCE / 'holdout_predictions.csv') if r['shipment_id'] == example['shipment_id'] and r['predictor'] == summary['predictor'])
    slices = read_csv(EVIDENCE / 'holdout_slices.csv')
    future_slices = {r['slice']: r for r in slices if r['predictor'] == summary['predictor'] and r['dimension'] == 'future_disruption_slice'}
    assert cfg['seed'] == 42 and cfg['n_shipments'] == 300 and len(cfg['routes']) == 5
    assert example['shipment_id'] == plan['holdout_ids'][0] == 'SHP-000241'
    assert summary['predictor'] == 'ridge_propagation_alpha_10'
    # Parameters in the compact mechanism table are checked, not inferred from fitted data.
    expected = {'departure': {'normal_mean_h': 2., 'normal_std_h': 4., 'early_bound_h': -2., 'disruption_probability': .1, 'disruption_min_h': 12., 'disruption_max_h': 36.},
                'voyage': {'weather_beta_a': 2., 'weather_beta_b': 5., 'weather_effect_h': 8., 'noise_std_fraction': .04, 'minimum_duration_fraction': .75},
                'port': {'congestion_beta_a': 2., 'congestion_beta_b': 3., 'base_multiplier': .75, 'congestion_multiplier': 1.1, 'noise_std_h': 2., 'minimum_duration_h': 1., 'disruption_probability': .1, 'disruption_min_h': 12., 'disruption_max_h': 36.},
                'inland': {'noise_std_fraction': .15, 'minimum_duration_h': 1., 'disruption_probability': .08, 'disruption_min_h': 4., 'disruption_max_h': 12.}}
    assert all(cfg[k] == v for k, v in expected.items())
    for name, filename in [('Report','LiberationSans-Regular.ttf'), ('Report-Bold','LiberationSans-Bold.ttf'), ('Report-Italic','LiberationSans-Italic.ttf')]:
        pdfmetrics.registerFont(TTFont(name, str(FONT_ROOT / filename)))
    pdfmetrics.registerFontFamily('Report', normal='Report', bold='Report-Bold', italic='Report-Italic', boldItalic='Report-Bold')
    styles = {
        'title': ParagraphStyle('title', fontName='Report-Bold', fontSize=20, leading=23, textColor=BLACK, spaceAfter=8),
        'subtitle': ParagraphStyle('subtitle', fontName='Report', fontSize=9.4, leading=12, textColor=BLACK, spaceAfter=12),
        'body': ParagraphStyle('body', fontName='Report', fontSize=10.1, leading=13.2, textColor=BLACK, spaceAfter=7),
        'small': ParagraphStyle('small', fontName='Report', fontSize=9.0, leading=11.5, textColor=BLACK, spaceAfter=6),
        'heading': ParagraphStyle('heading', fontName='Report-Bold', fontSize=11.4, leading=14, textColor=BLACK, spaceBefore=8, spaceAfter=6, keepWithNext=True),
        'formula': ParagraphStyle('formula', fontName='Report', fontSize=10.3, leading=14.3, textColor=BLACK, spaceAfter=8),
        'cell': ParagraphStyle('cell', fontName='Report', fontSize=9.0, leading=11.6, textColor=BLACK),
        'number': ParagraphStyle('number', fontName='Report', fontSize=9.0, leading=11.6, textColor=BLACK, alignment=TA_RIGHT),
        'cellhead': ParagraphStyle('cellhead', fontName='Report-Bold', fontSize=9.0, leading=11.6, textColor=BLACK),
        'ref': ParagraphStyle('ref', fontName='Report', fontSize=8.3, leading=10.8, textColor=BLACK, spaceAfter=0),
    }
    story, md = [], ['# Shipment ETA and Delay Propagation\n']
    def text(s, style='body', markup=False):
        story.append(Paragraph(s if markup else escape(s), styles[style]))
        md.append(s+'\n')
    def heading(s):
        story.append(Paragraph(escape(s), styles['heading'])); md.append('## '+s+'\n')
    def title(s, subtitle):
        text(s,'title');text(subtitle,'subtitle')
    def table(headers, rows, widths, numeric=()):
        assert abs(sum(widths)-CONTENT_WIDTH)<1e-6
        values=[[Paragraph(escape(str(x)),styles['cellhead']) for x in headers]]
        values.extend([Paragraph(escape(str(x)), styles['number'] if i in numeric else styles['cell']) for i,x in enumerate(row)] for row in rows)
        t=Table(values,colWidths=widths,hAlign='LEFT',repeatRows=1)
        t.setStyle(TableStyle([('GRID',(0,0),(-1,-1),.45,BLACK),('LINEBELOW',(0,0),(-1,0),.8,BLACK),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4)]))
        story.extend([t,Spacer(1,8)])
        md.extend(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |',*['| '+' | '.join(map(str,r))+' |' for r in rows],''])
    def next_page():story.append(PageBreak());md.append('\n---\n')

    title('Shipment ETA and Delay Propagation','Sai Manoj | Aiqon technical assignment | Destination-port arrival checkpoint')
    text('<b>Purpose.</b> Support downstream delivery planning by estimating final ETA and delay magnitude from information available when a shipment reaches its destination port. The implementation combines a synthetic journey simulator, chronological evaluation and exact explanations of a small regression model.',markup=True)
    heading('1. Prediction contract')
    text('Let A₁ be actual port arrival, A₃ actual delivery and S₃ scheduled delivery. The target r is remaining transit time; r<sub rise="2.2" size="7.2">raw</sub> is the model output. All durations are hours and timestamps are UTC.',markup=True)
    text('r = A₃ - A₁<br/>ETA = A₁ + max(0, r<sub rise="2.2" size="7.2">raw</sub>)<br/>Signed deviation = ETA - S₃;  delay magnitude = max(0, ETA - S₃)','formula',True)
    text('A 20-column input allowlist admits schedules, route attributes, observed departure/arrival, completed-voyage weather and current congestion. Actual future readiness, dispatch, delivery, processing durations and disruption flags are excluded. Negative raw remaining time is clipped and flagged; no numerical delay probability is claimed.')
    heading('2. Synthetic journeys and ground truth')
    text(f"The dataset contains {audit['n_shipments']} shipments on five fictional routes, {min(audit['route_counts'].values())} per route. Planned voyage V, port processing P, inland travel L, buffer B and dispatch interval H define S₁ = S₀ + V, S₂ = S₁ + P and S₃ = S₂ + L + B. Handling and customs form one aggregate readiness stage.")
    table(['Stage','Implemented mechanism'],[
        ['Departure','A₀ = S₀ + max(-2, N(2, 4)) + Bern(0.10) × U(12, 36).'],
        ['Voyage','W ~ Beta(2, 5); duration = max(0.75V, V + 8W + N(0, 0.04V)); A₁ = A₀ + duration.'],
        ['Port / customs','C ~ Beta(2, 3); processing = max(1, P(0.75 + 1.10C) + N(0, 2) + Bern(0.10) × U(12, 36)); A₂ = A₁ + processing.'],
        ['Dispatch / inland','Dispatch at the first S₂ + kH ≥ A₂, integer k ≥ 0. Inland duration = max(1, L + N(0, 0.15L) + Bern(0.08) × U(4, 12)); A₃ = dispatch + inland duration.'],
    ],[88,CONTENT_WIDTH-88])
    text('N(mean, standard deviation), U(uniform) and Bern(Bernoulli) use independent draws unless linked above. Seed 42, SeedSequence child 0, PCG64 and a fixed draw order make generation reproducible with the recorded versions. Integer-nanosecond dispatch arithmetic includes exact-slot readiness. These parameters are assumptions, not measured port statistics. Congestion has no reporting lag; shipments do not share capacity.','small')
    heading('3. Delay propagation and recovery')
    text('For S₁ = 0, P = 12, L = 8, B = 8 and H = 6, the deadline is S₃ = 28. Arrival at A₁ = 6 with 8 h processing gives dispatch at 18 and delivery at 26: 2 h early. With 20 h processing, dispatch is 30 and delivery 38: 10 h late. Buffers absorb lateness; missed slots amplify it without double-counting upstream delays.')
    text(f"The generated audit contains {audit['arrival_counts']['early']} early and {audit['arrival_counts']['late']} late deliveries; {audit['late_port_but_not_late_delivery']} late port arrivals recover by delivery and {audit['late_port_and_amplified_final_lateness']} amplify final lateness. Full equations, parameters and schemas are in docs/DATA_SPECIFICATION.md and config/simulation.toml.",'small')

    next_page()
    title('Model and evaluation design','Time-aware selection | Explicit label availability | Reconstructable explanations')
    heading('4. Features and model choice')
    text('The base model uses planned processing, planned inland duration and observed congestion. The nine-feature propagation model adds signed port-arrival deviation, time until first dispatch, dispatch interval, congestion × planned processing, and sine/cosine dispatch phase. Route IDs are audit labels; absolute dates and future outcomes are not features.')
    text('Each scikit-learn pipeline applies deterministic features, training-only StandardScaler and Ridge(SVD) [1, 2]. Eight candidates span two feature groups and α values {0.1, 1, 10, 100}. Selection minimizes pooled validation ETA MAE; within 0.05 h of the minimum, it prefers fewer features, then larger α. Ridge is recommended only if it beats the best validation baseline by more than 0.05 h.')
    gap=selection['group_best_validation_mae_h']['base']-selection['group_best_validation_mae_h']['propagation']
    text(f"Propagation Ridge (α = 10) was selected. Base versus propagation MAE was {selection['group_best_validation_mae_h']['base']:.3f} versus {selection['group_best_validation_mae_h']['propagation']:.3f} h: a small {gap:.3f} h ({gap*60:.2f} min) gain. Ridge limits complexity and supports exact explanations, but cannot fully represent dispatch discontinuities. Squared-error fitting and MAE selection optimize different quantities.")
    heading('5. Chronological evaluation')
    table(['Partition','Training','Validation / holdout','Unavailable labels'],[
        *[[f"Validation {f['fold']}",len(f['training_ids']),len(f['validation_ids']),len(f['excluded_unavailable_label_ids'])] for f in plan['folds']],
        ['Final fit / holdout',len(plan['development_ids']),len(plan['holdout_ids']),len(plan['excluded_unavailable_label_ids'])],
    ],[143,78,145,CONTENT_WIDTH-366],numeric=(1,2,3))
    text('Records are ordered by port-arrival checkpoint. Training requires delivery strictly before the next window starts; three of the 240 earlier shipments lack available labels at the final cutoff. The three validation windows contain 120 distinct records. The selected model was fitted on 237 shipments, then held fixed across 60 holdout records. Data, IDs, model, source and metric definitions were frozen before scoring.','small')
    heading('6. Per-shipment explanation')
    text('For feature x<sub rise="2.2" size="7.2">j</sub>, fitted training mean μ<sub rise="2.2" size="7.2">j</sub>, scale s<sub rise="2.2" size="7.2">j</sub> and standardized coefficient β<sub rise="2.2" size="7.2">j</sub>, the contribution is c<sub rise="2.2" size="7.2">j</sub> = β<sub rise="2.2" size="7.2">j</sub>(x<sub rise="2.2" size="7.2">j</sub> - μ<sub rise="2.2" size="7.2">j</sub>)/s<sub rise="2.2" size="7.2">j</sub>. The intercept plus Σ<sub rise="2.2" size="7.2">j</sub>c<sub rise="2.2" size="7.2">j</sub> equals raw remaining time. This training-mean reference need not describe a feasible shipment; contributions explain model arithmetic, not causal stage delays [3].',markup=True)
    table([f"First holdout checkpoint: {example['shipment_id']}",'Contribution / hours'],[
        ['Training-mean reference',f"{example['reference_remaining_h']:.3f}"],
        *[[g['label'],f"{g['contribution_h']:+.3f}"] for g in example['group_contributions']],
        ['Predicted remaining time; no duration clipping',f"{example['predicted_remaining_h']:.3f}"],
    ],[CONTENT_WIDTH-118,118],numeric=(1,))
    deadline=-example['deadline_offset_h']
    text(f"Port processing/congestion is the largest positive group relative to the reference. The deadline is {deadline:.3f} h after arrival; predicted delay is {example['predicted_delay_magnitude_h']:.3f} h and ETA is 9 May 2025, 08:50:40 UTC. Actual remaining time is {float(case['actual_remaining_h']):.3f} h: ETA error is {float(case['eta_error_h']):.3f} h. This earliest holdout record was not chosen for accuracy. Display values are rounded.",'small')
    text(f"Duration clipping, deadline offset and delay flooring are reported separately. Maximum holdout reconstruction error was {summary['max_explanation_reconstruction_error_h']:.2e} h; exact reconstruction is numerical consistency, not prediction accuracy.",'small')

    next_page()
    title('Results and engineering assessment','Frozen synthetic holdout | Same 60 shipments for every predictor')
    heading('7. Baselines and measured results')
    text('Scheduled ETA = max(A₁, S₃). Carry-forward ETA = max(A₁, S₃ + A₁ - S₁); its remaining time equals planned processing + inland duration + buffer. The comparison baseline was selected on validation. All table values below are hours; lower error is better.')
    rows=[]
    for name,label in [('scheduled','Scheduled'),('carry_forward','Carry-forward'),(summary['predictor'],'Selected Ridge')]:
        m=metrics[name]
        val=selection['selected_ridge']['validation_eta_mae_h'] if name==summary['predictor'] else float(baseline_validation[name]['eta_mae_h'])
        rows.append([label,f'{val:.3f}',f"{m['eta_mae_h']:.3f}",f"{m['eta_rmse_h']:.3f}",f"{m['delay_mae_h']:.3f}"])
    table(['Predictor','Validation ETA MAE','Holdout ETA MAE','Holdout ETA RMSE','Holdout delay MAE'],rows,[113,96,96,96,CONTENT_WIDTH-401],numeric=(1,2,3,4))
    text('Validation n = 120; holdout n = 60. With e = predicted ETA - actual delivery, MAE = mean(|e|), RMSE = √mean(e²), and bias = mean(e). Delay MAE compares nonnegative predicted and actual delay. Ridge and carry-forward clipped zero durations; scheduled ETA clipped one.','small')
    pair=summary['paired_vs_validation_baseline']; m=metrics[summary['predictor']]
    text(f"<b>The validation advantage did not persist on holdout.</b> Ridge MAE is {-pair['mean_absolute_error_improvement_h']:.3f} h ({-pair['mean_absolute_error_improvement_h']*60:.2f} min) worse than carry-forward, with smaller absolute error on {pair['better']} shipments and larger error on {pair['worse']}. RMSE, delay MAE and tail error are also worse. The preselected model and all comparisons are retained; learned-model superiority is not established.",markup=True)
    heading('8. Uncertainty and failure analysis')
    ci=summary['bootstrap_intervals']; model_ci=ci['selected_eta_mae_h']; pair_ci=ci['paired_eta_mae_improvement_h']
    text(f"The predeclared paired percentile bootstrap uses {ci['resamples']:,} shipment resamples and PCG64 seed {ci['seed']} [4]. Approximate 95% intervals are [{model_ci['lower_h']:.3f}, {model_ci['upper_h']:.3f}] h for Ridge MAE and [{pair_ci['lower_h']:.3f}, {pair_ci['upper_h']:.3f}] h for carry-forward minus Ridge MAE. Spanning zero establishes neither superiority nor equivalence. Intervals condition on the fitted model, assume independently resampleable shipments and exclude training/selection variability; they are not individual ETA intervals.")
    text(f"Ridge bias is {m['eta_bias_h']:.3f} h, P95 absolute error {m['eta_p95_abs_error_h']:.3f} h and maximum error {m['eta_max_abs_error_h']:.3f} h. Port-only future-disruption cases have MAE {float(future_slices['port_only']['eta_mae_h']):.3f} h (n = {future_slices['port_only']['n']}); cases with both port and inland disruptions have {float(future_slices['both']['eta_mae_h']):.3f} h (n = {future_slices['both']['n']}). These small, outcome-defined slices diagnose errors; their future flags never enter prediction.")
    text(f"Across {review['stress_scenario_rows']:,} controlled development response rows, no directional decreases occurred in the declared scenarios; this is a behavior check, not accuracy or global monotonicity. All {review['invalid_inputs_rejected']} declared invalid-input cases were rejected. {summary['shipments_with_feature_range_flags']} holdout shipments had feature-range flags, which are not confidence scores.")
    text('Missing inputs are rejected, not imputed. Unseen-route accuracy, shared queues, reporting lags and operational drift are unverified. Independent real data and a new evaluation are needed before deployment or further model selection; the exposed holdout is not an optimization target.')
    heading('9. Verification and reproducibility')
    text(f"All {verification['tests_passed']} tests passed again with warnings as errors on Linux/Python 3.12.14. The seven CLI stages and explicit CSV prediction completed; dataset hashes and the final summary matched the original run. Independent calculations passed {verification['independent_recorded_checks']} recorded checks. The supplied Windows/Python 3.12.1 clean-environment run also passed 186 tests and reproduced the workflow. These are repeatability checks, not new independent holdout evaluations.",'small')
    text('README.md provides setup and execution. reports/FINAL_AUDIT.md records verification scope; reports/evidence retains numerical inputs. The report builder and pinned optional dependencies are in report_tools. No feature, model, simulation or evaluation setting changed after holdout exposure.','small')
    refs=[('1. Ridge','https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html'),('2. Leakage prevention','https://scikit-learn.org/stable/common_pitfalls.html'),('3. Linear coefficients','https://scikit-learn.org/stable/auto_examples/inspection/plot_linear_model_coefficient_interpretation.html'),('4. Bootstrap methods','https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html')]
    links=[f'<link href="{url}" color="black">{escape(label)}</link>' for label,url in refs]
    text('References: '+' | '.join(links),'ref',True)
    def page(canvas,doc):
        canvas.saveState(); canvas.setFillGray(0);canvas.setStrokeGray(0)
        canvas.setFont('Report',8.5)
        canvas.drawString(MARGIN,H-30,'AIQON / TECHNICAL REPORT')
        canvas.drawRightString(W-MARGIN,H-30,'Synthetic study · Reproducible evaluation')
        canvas.setLineWidth(.5); canvas.line(MARGIN,38,W-MARGIN,38)
        canvas.drawString(MARGIN,25,'Shipment ETA and Delay Propagation')
        canvas.drawRightString(W-MARGIN,25,f'{doc.page} / 3')
        canvas.restoreState()
    doc=BaseDocTemplate(str(output),pagesize=A4,leftMargin=MARGIN,rightMargin=MARGIN,topMargin=H-TOP,bottomMargin=BOTTOM,title='Shipment ETA and Delay Propagation',author='Sai Manoj',subject='Approach, assumptions, measured results and reproducibility',invariant=1,initialFontName='Report')
    frame=Frame(MARGIN,BOTTOM,CONTENT_WIDTH,TOP-BOTTOM,leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0,id='body')
    doc.addPageTemplates(PageTemplate(id='report',frames=[frame],onPage=page))
    output.parent.mkdir(parents=True,exist_ok=True)
    doc.build(story,canvasmaker=Canvas)
    # A readable companion; PDF-specific markup is preserved as supported Markdown HTML.
    (ROOT/'reports/technical_report.md').write_text('\n'.join(md).rstrip('\n')+'\n',encoding='utf-8')
    return output

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'reports/Shipment_ETA_Technical_Report.pdf')
    args=parser.parse_args()
    print(build(args.output.resolve()))
