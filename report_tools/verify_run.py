"""Independent numerical audit of the fixed assignment workflow.

Run after all seven default CLI commands. Uses only NumPy/pandas and stdlib;
does not import the project's metric or model implementations. Checks the
preserved reference evidence and refuses to overwrite an existing audit.
This reproduces an exposed experiment; it is not a new holdout evaluation.
"""
from pathlib import Path
import json, hashlib, math, tomllib, sys
from importlib.metadata import version
import numpy as np
import pandas as pd
import argparse
parser = argparse.ArgumentParser(description='Independently audit the fixed assignment run without importing project model/evaluation functions.')
parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[1])
parser.add_argument('--output', type=Path, default=None)
args = parser.parse_args()
p = args.project_root.resolve()
output = args.output or p / 'artifacts/numerical_verification.json'
if output.exists():
    raise SystemExit('Output already exists; choose a new --output path to preserve evidence.')
if not __debug__:
    raise SystemExit('Run without Python -O: this audit requires assertions.')
checks = []

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def check(name, actual, expected, tolerance=1e-09):
    if isinstance(actual, (float, np.floating)):
        ok = math.isclose(float(actual), float(expected), rel_tol=0, abs_tol=tolerance)
        actual, expected = (float(actual), float(expected))
    else:
        ok = actual == expected
    if not ok:
        raise AssertionError((name, actual, expected))
    checks.append({'claim': name, 'actual': actual, 'expected': expected, 'passed': True})

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
evidence_manifest = read(p / 'reports/evidence/evidence_manifest.json')
for item in evidence_manifest['files']:
    assert digest(p / 'reports/evidence' / item['file']) == item['sha256']
for name, h in evidence_manifest['dataset_csv_sha256'].items():
    assert digest(p / 'data/generated' / name) == h
for name, h in evidence_manifest['frozen_executable_sha256'].items():
    check('frozen:' + name, digest(p / name), h)
for line in (p / 'requirements.txt').read_text().splitlines():
    name, pin = line.split('==')
    check('package:' + name, version(name), pin)
journeys = pd.read_csv(p / 'data/generated/journeys.csv')
inputs = pd.read_csv(p / 'data/generated/checkpoint_inputs.csv')
for frame in (journeys, inputs):
    for c in frame:
        if c.endswith('_utc'):
            frame[c] = pd.to_datetime(frame[c], utc=True, format='ISO8601').dt.as_unit('ns')
check('shipment_count', len(journeys), 300)
check('unique_shipments', journeys.shipment_id.nunique(), 300)
check('route_counts', sorted(journeys.route_id.value_counts().tolist()), [60] * 5)
check('input_columns', len(inputs.columns), 20)
check('early', int((journeys.signed_arrival_deviation_h < 0).sum()), 89)
check('late', int((journeys.signed_arrival_deviation_h > 0).sum()), 211)
portdev = (journeys.actual_port_arrival_utc - journeys.scheduled_port_arrival_utc).dt.total_seconds() / 3600
check('recovery', int(((portdev > 0) & (journeys.signed_arrival_deviation_h <= 0)).sum()), 53)
check('amplification', int(((portdev > 0) & (journeys.signed_arrival_deviation_h > portdev)).sum()), 110)
J = journeys.set_index('shipment_id')
max_remaining = max_signed = 0.0
for row in journeys.itertuples(index=False):
    step = round(row.dispatch_interval_h * 3600000000000)
    delta = row.actual_ready_utc.value - row.scheduled_ready_utc.value
    dispatch = row.scheduled_ready_utc.value + max(0, -(-delta // step)) * step
    assert dispatch == row.actual_dispatch_utc.value
    remaining = (row.actual_delivery_utc.value - row.actual_port_arrival_utc.value) / 3600000000000
    signed = (row.actual_delivery_utc.value - row.scheduled_delivery_utc.value) / 3600000000000
    max_remaining = max(max_remaining, abs(row.remaining_h - remaining))
    max_signed = max(max_signed, abs(row.signed_arrival_deviation_h - signed))
    assert abs(row.delay_magnitude_h - max(0, signed)) < 1e-09
check('independent_remaining_max_difference_h', max_remaining, 0.0)
check('independent_signed_max_difference_h', max_signed, 0.0)
plan = read(p / 'artifacts/stage5/split_manifest.json')
check('development_n', len(plan['development_ids']), 237)
check('holdout_n', len(plan['holdout_ids']), 60)
check('excluded_n', len(plan['excluded_unavailable_label_ids']), 3)
check('fold_training_n', [len(f['training_ids']) for f in plan['folds']], [114, 153, 194])
check('fold_validation_n', [len(f['validation_ids']) for f in plan['folds']], [40, 40, 40])
check('fold_excluded_n', [len(f['excluded_unavailable_label_ids']) for f in plan['folds']], [3, 4, 3])
for f in plan['folds']:
    assert (J.loc[f['training_ids'], 'actual_delivery_utc'] < pd.Timestamp(f['cutoff_utc'])).all()
    assert not set(f['training_ids']) & set(f['validation_ids'])
    assert not set(f['training_ids'] + f['validation_ids']) & set(plan['holdout_ids'])
assert (J.loc[plan['development_ids'], 'actual_delivery_utc'] < pd.Timestamp(plan['holdout_cutoff_utc'])).all()
for stage, metricfile, predfile, idcol in [('stage5', 'baseline_metrics.csv', 'validation_predictions.csv', 'baseline'), ('stage6', 'candidate_metrics.csv', 'candidate_validation_predictions.csv', 'candidate'), ('stage9_final', 'metrics.csv', 'holdout_predictions.csv', 'predictor')]:
    m = pd.read_csv(p / 'artifacts' / stage / metricfile)
    pred = pd.read_csv(p / 'artifacts' / stage / predfile)
    pred['predicted_eta_utc'] = pd.to_datetime(pred.predicted_eta_utc, utc=True, format='ISO8601').dt.as_unit('ns')
    for record in m.to_dict('records'):
        r = pred.loc[pred[idcol] == record[idcol]].copy()
        if record['scope'].startswith('fold_'):
            r = r.loc[r.fold == int(record['scope'].split('_')[-1])]
        r = r.set_index('shipment_id')
        truth = J.loc[r.index]
        errors = (r.predicted_eta_utc - truth.actual_delivery_utc).dt.total_seconds() / 3600
        delay = (r.predicted_eta_utc - truth.scheduled_delivery_utc).dt.total_seconds().div(3600).clip(lower=0)
        assert np.allclose(delay, r.predicted_delay_magnitude_h, atol=1e-09, rtol=0)
        assert np.allclose(errors, r.eta_error_h, atol=1e-09, rtol=0)
        observed = {'n': len(r), 'eta_mae_h': errors.abs().mean(), 'eta_rmse_h': np.sqrt(np.mean(errors ** 2)), 'eta_bias_h': errors.mean(), 'delay_mae_h': (delay - truth.delay_magnitude_h).abs().mean(), 'clipped_count': int(r.was_clipped.sum()), 'eta_p95_abs_error_h': errors.abs().quantile(0.95), 'eta_max_abs_error_h': errors.abs().max()}
        for key, val in observed.items():
            if key in record:
                check(stage + ':' + record[idcol] + ':' + record['scope'] + ':' + key, float(val), float(record[key]))
summary = read(p / 'artifacts/stage9_final/summary.json')
preds = pd.read_csv(p / 'artifacts/stage9_final/holdout_predictions.csv')
selected = preds.loc[preds.predictor == summary['predictor']].set_index('shipment_id')
baseline = preds.loc[preds.predictor == 'carry_forward'].set_index('shipment_id').loc[selected.index]
difference = baseline.eta_error_h.abs() - selected.eta_error_h.abs()
check('paired_better', int((difference > 1e-08).sum()), 29)
check('paired_worse', int((difference < -1e-08).sum()), 31)
check('paired_improvement', difference.mean(), summary['paired_vs_validation_baseline']['mean_absolute_error_improvement_h'])
rng = np.random.Generator(np.random.PCG64(2026))
idx = rng.integers(0, 60, size=(5000, 60))
for name, samples in [('selected_eta_mae_h', selected.eta_error_h.abs().to_numpy()[idx].mean(1)), ('paired_eta_mae_improvement_h', difference.to_numpy()[idx].mean(1))]:
    bounds = np.quantile(samples, [0.025, 0.975], method='linear')
    for key, val in zip(('lower_h', 'upper_h'), bounds):
        check('bootstrap:' + name + ':' + key, val, summary['bootstrap_intervals'][name][key])
expl = read(p / 'artifacts/stage9_final/holdout_explanations.json')['records']
max_diff = 0.0
for row in expl:
    terms = row['feature_terms']
    for term in terms:
        assert abs(term['coefficient_standardized_h'] * (term['value'] - term['reference_value']) / term['scale'] - term['contribution_h']) < 1e-10
    reconstructed = row['reference_remaining_h'] + sum((t['contribution_h'] for t in terms))
    max_diff = max(max_diff, abs(reconstructed - row['raw_remaining_h']))
    assert abs(sum((g['contribution_h'] for g in row['group_contributions'])) - sum((t['contribution_h'] for t in terms))) < 1e-10
check('explanation_max_reconstruction_h', max_diff, summary['max_explanation_reconstruction_error_h'], 1e-12)
check('explanation_first_id', expl[0]['shipment_id'], plan['holdout_ids'][0])
for dim in ['route_id', 'congestion_slice', 'arrival_slice', 'future_disruption_slice']:
    sl = pd.read_csv(p / 'artifacts/stage9_final/slice_metrics.csv')
    for metric in sl.loc[sl.dimension == dim].to_dict('records'):
        r = preds.loc[(preds.predictor == metric['predictor']) & (preds[dim] == metric['slice'])]
        check('slice:' + metric['predictor'] + ':' + dim + ':' + metric['slice'] + ':n', len(r), metric['n'])
        if len(r):
            check('slice:' + metric['predictor'] + ':' + dim + ':' + metric['slice'] + ':mae', r.eta_error_h.abs().mean(), metric['eta_mae_h'])
check('old_summary_exact', summary, read(p / 'reports/evidence/holdout_summary.json'))
result = {'checked_claims': len(checks), 'all_passed': True, 'metric_tolerance_h': 1e-09, 'explanation_tolerance_h': 1e-10, 'checks': checks}
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print('Independent evidence checks passed:', len(checks))
