"""Score the reviewed, unchanged predictor on the frozen final holdout."""

from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from .dataset_io import read_table, sha256
from .evaluation import BASELINES, baseline_prediction, prediction_errors, summarize_errors
from .evaluation_freeze import FINAL_PROTOCOL, freeze_context, read_json, verify_bundle, write_json
from .explanations import predict_and_explain
from .robustness import add_slice_context, paired_comparison, slice_metrics


def _select(frame, ids):
    return frame.set_index("shipment_id").loc[ids].reset_index()


def paired_bootstrap(selected_errors, baseline_errors):
    """Paired percentile intervals conditional on this fitted model and sample."""
    if not isinstance(selected_errors, pd.Series) or not isinstance(baseline_errors, pd.Series):
        raise ValueError("Bootstrap errors must be Series indexed by shipment ID")
    if (len(selected_errors) < 2 or not selected_errors.index.is_unique or not baseline_errors.index.is_unique
            or set(selected_errors.index) != set(baseline_errors.index)):
        raise ValueError("Bootstrap requires matching unique IDs and at least two shipments")
    for values in [selected_errors, baseline_errors]:
        if not pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values) or not np.isfinite(values.to_numpy()).all():
            raise ValueError("Bootstrap errors must be finite numbers")
    selected = np.abs(selected_errors.to_numpy(dtype=float))
    baseline = np.abs(baseline_errors.loc[selected_errors.index].to_numpy(dtype=float))
    rng = np.random.Generator(np.random.PCG64(FINAL_PROTOCOL["bootstrap_seed"]))
    indices = rng.integers(0, len(selected), size=(FINAL_PROTOCOL["bootstrap_resamples"], len(selected)))
    model_means = selected[indices].mean(axis=1)
    improvement_means = (baseline - selected)[indices].mean(axis=1)

    def interval(samples, estimate):
        bounds = np.quantile(samples, [0.025, 0.975], method="linear")
        return {"estimate_h": float(estimate), "lower_h": float(bounds[0]), "upper_h": float(bounds[1])}

    return {"n": len(selected), "confidence_level": .95, "method": "paired_shipments_percentile",
            "resamples": FINAL_PROTOCOL["bootstrap_resamples"], "seed": FINAL_PROTOCOL["bootstrap_seed"],
            "selected_eta_mae_h": interval(model_means, selected.mean()),
            "paired_eta_mae_improvement_h": interval(improvement_means, (baseline - selected).mean()),
            "interpretation": "Approximate intervals conditional on the fitted model and observed sample; assumes independently resampleable shipments. Not individual ETA intervals; training/selection variability is excluded."}


def final_metrics(errors):
    return {**summarize_errors(errors), "eta_p95_abs_error_h": float(errors.eta_error_h.abs().quantile(.95)),
            "eta_max_abs_error_h": float(errors.eta_error_h.abs().max())}


def plot_final_results(metrics, predictions, predictor, path):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    figure = Figure(figsize=(11, 4.8), layout="constrained")
    FigureCanvasAgg(figure)
    left, right = figure.subplots(1, 2)
    labels = ["Selected predictor" if name == predictor else name.replace("_", " ") for name in metrics.predictor]
    bars = left.bar(labels, metrics.eta_mae_h, color=["#20689c" if name == predictor else "#7b8997" for name in metrics.predictor])
    left.bar_label(bars, labels=[f"{value:.3f}" for value in metrics.eta_mae_h], padding=4)
    left.set_ylim(0, left.get_ylim()[1] * 1.22)
    left.set_ylabel("Holdout ETA MAE (hours; lower is better)")
    left.set_title("Frozen predictors on the same holdout")
    selected = predictions.loc[predictions.predictor == predictor]
    right.scatter(selected.actual_remaining_h, selected.predicted_remaining_h, s=27, alpha=.75, color="#20689c")
    maximum = float(max(selected.actual_remaining_h.max(), selected.predicted_remaining_h.max()) * 1.05)
    right.plot([0, maximum], [0, maximum], "--", color="#555555", linewidth=1)
    right.set_xlim(0, maximum)
    right.set_ylim(0, maximum)
    right.set_xlabel("Actual remaining hours")
    right.set_ylabel("Predicted remaining hours")
    right.set_title(f"Selected predictor: {len(selected)} holdout shipments")
    figure.supxlabel("Synthetic holdout evaluation; no refitting or model selection from these scores.", fontsize=10)
    figure.savefig(path, dpi=150)


def render_final_report(summary, metrics, slices):
    lines = ["# Final synthetic holdout evaluation", "",
             f"Predictor fixed before scoring: `{summary['predictor']}`. "
             f"Training shipments: {summary['training_shipments']}; holdout shipments: {summary['holdout_shipments']}.", "",
             "All predictors use the same checkpoint inputs and holdout IDs. No model was refitted or selected using these results.", "",
             "| Predictor | n | ETA MAE | ETA RMSE | ETA bias | Delay MAE | P95 absolute ETA error | Max absolute ETA error | Clipped |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in metrics.itertuples():
        lines.append(f"| {row.predictor} | {row.n} | {row.eta_mae_h:.6f} | {row.eta_rmse_h:.6f} | "
                     f"{row.eta_bias_h:.6f} | {row.delay_mae_h:.6f} | {row.eta_p95_abs_error_h:.6f} | "
                     f"{row.eta_max_abs_error_h:.6f} | {row.clipped_count} |")
    pair = summary["paired_vs_validation_baseline"]
    lines.extend(["", "All error values are hours. Positive bias means predicting delivery later than actual delivery.", "",
                  f"Compared with the validation-selected baseline `{pair['baseline']}`: "
                  f"{pair['better']} smaller absolute errors, {pair['worse']} larger, {pair['tied']} tied. "
                  f"Mean paired improvement: {pair['mean_absolute_error_improvement_h']:.6f} h. Positive favors the selected predictor.", "",
                  "## Conditional resampling intervals", ""])
    for name in ["selected_eta_mae_h", "paired_eta_mae_improvement_h"]:
        value = summary["bootstrap_intervals"][name]
        lines.append(f"- {name}: estimate {value['estimate_h']:.6f} h; 95% percentile interval "
                     f"[{value['lower_h']:.6f}, {value['upper_h']:.6f}] h.")
    lines.extend(["", summary["bootstrap_intervals"]["interpretation"], "",
                  "Intervals use 5,000 paired resamples with PCG64 seed 2026. They do not establish real-world coverage "
                  "or account for unseen shared disruptions. An improvement interval spanning zero does not establish a consistent advantage.", "",
                  "## Selected predictor slices", "",
                  "| Dimension | Slice | n | ETA MAE (h) | Bias (h) | Small sample |",
                  "| --- | --- | ---: | ---: | ---: | ---: |"])
    for row in slices.loc[slices.predictor == summary["predictor"]].itertuples():
        mae = "n/a" if pd.isna(row.eta_mae_h) else f"{row.eta_mae_h:.3f}"
        bias = "n/a" if pd.isna(row.eta_bias_h) else f"{row.eta_bias_h:.3f}"
        lines.append(f"| {row.dimension} | {row.slice} | {row.n} | {mae} | {bias} | {row.small_sample} |")
    lines.extend(["", "Slices are descriptive, overlap across dimensions and are not used for tuning. n<10 is a display warning, "
                  "not a statistical sufficiency guarantee. Future disruption flags are outcome-only diagnostic labels.", "",
                  f"Selected-model holdout feature-range flags: {summary['shipments_with_feature_range_flags']}. "
                  f"Maximum explanation reconstruction error: {summary['max_explanation_reconstruction_error_h']:.3g} h.", "",
                  "## Limitations and next actions", "",
                  "This is one 60-shipment sample from an illustrative synthetic mechanism, not real-port validation. "
                  "Future random disruptions remain unobserved at the checkpoint. Preserve large errors and all comparison results. "
                  "The holdout is now exposed: do not tune against it or call later changes independently tested on the same data. "
                  "Complete environment reproduction, documentation and submission review before presenting this as a finished assignment.", ""])
    return "\n".join(lines)


def evaluate_holdout(dataset_dir, stage5_dir, model_dir, review_dir, freeze_dir, output_dir):
    """Score only after verifying the saved freeze; never fit or select a model."""
    output_dir, freeze_dir = Path(output_dir).resolve(), Path(freeze_dir)
    if output_dir.exists():
        raise FileExistsError(f"Output already exists: {output_dir}. Preserve the original final evaluation.")
    verify_bundle(freeze_dir, {"freeze.json"})
    frozen = read_json(freeze_dir / "freeze.json")
    current, inputs, targets, model, metadata = freeze_context(dataset_dir, stage5_dir, model_dir, review_dir)
    if frozen != current:
        raise ValueError("Evaluation freeze no longer matches the source, protocol, model or data; do not bypass this check")
    ids = frozen["holdout_ids"]
    if set(ids).intersection(frozen["training_ids"]) or len(set(ids)) != len(ids):
        raise ValueError("Holdout IDs must be unique and separate from training")
    x, y = _select(inputs, ids), _select(targets, ids)
    selected, explanations = predict_and_explain(model, metadata, x)
    predictor = frozen["predictor"]
    predictions, scores = [], []
    for name in dict.fromkeys([predictor, *BASELINES]):
        served = selected if name == predictor else baseline_prediction(x, name)
        errors = prediction_errors(x, y, served)
        scores.append({"predictor": name, "scope": "final_holdout", **final_metrics(errors)})
        table = served.join(errors.drop(columns="was_clipped")).reset_index()
        table["predictor"] = name
        predictions.append(table)
    predicted = pd.concat(predictions, ignore_index=True)
    predicted = add_slice_context(predicted, inputs, read_table(Path(dataset_dir) / "journeys.csv"))
    actuals = y[["shipment_id", "remaining_h", "delay_magnitude_h"]].rename(
        columns={"remaining_h": "actual_remaining_h", "delay_magnitude_h": "actual_delay_magnitude_h"})
    predicted = predicted.merge(actuals, on="shipment_id", validate="many_to_one")
    predicted = predicted.merge(x[["shipment_id", "as_of_utc", "scheduled_delivery_utc"]], on="shipment_id", validate="many_to_one")
    metrics, slices = pd.DataFrame(scores), slice_metrics(predicted)
    comparison = frozen["comparison_baseline"]
    paired = paired_comparison(predicted, predictor, comparison)
    selected_errors = predicted.loc[predicted.predictor == predictor].set_index("shipment_id").eta_error_h
    baseline_errors = predicted.loc[predicted.predictor == comparison].set_index("shipment_id").eta_error_h
    bootstrap = paired_bootstrap(selected_errors, baseline_errors)
    largest = predicted.loc[predicted.predictor == predictor].copy()
    largest["absolute_eta_error_h"] = largest.eta_error_h.abs()
    largest = largest.sort_values(["absolute_eta_error_h", "shipment_id"], ascending=[False, True], kind="stable").head(10)
    records = explanations["records"]
    explanations["evaluation_scope"] = "final_holdout"
    summary = {"schema_version": 1, "evaluation_scope": "final_holdout", "predictor": predictor,
               "training_shipments": len(frozen["training_ids"]), "holdout_shipments": len(ids),
               "excluded_unavailable_labels": len(frozen["excluded_unavailable_label_ids"]),
               "holdout_scored": True, "model_refitted": False, "predictor_changed": False,
               "metrics": scores, "paired_vs_validation_baseline": paired, "bootstrap_intervals": bootstrap,
               "shipments_with_feature_range_flags": sum(bool(record["training_range_flags"]) for record in records),
               "max_explanation_reconstruction_error_h": max(record["reconstruction_abs_error_h"] for record in records),
               "interpretation": "Final evaluation of the frozen predictor on the reserved synthetic holdout; no post-holdout tuning."}
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".final-build-", dir=output_dir.parent) as temporary:
        bundle = Path(temporary) / "bundle"
        bundle.mkdir()
        for name, frame in {"holdout_predictions.csv": predicted, "metrics.csv": metrics,
                            "slice_metrics.csv": slices, "largest_errors.csv": largest}.items():
            exported = frame.copy()
            for column in exported:
                if isinstance(exported[column].dtype, pd.DatetimeTZDtype):
                    exported[column] = exported[column].map(lambda value: value.isoformat())
            exported.to_csv(bundle / name, index=False, float_format="%.17g", lineterminator="\n")
        write_json(bundle / "summary.json", summary)
        write_json(bundle / "bootstrap_intervals.json", bootstrap)
        write_json(bundle / "holdout_explanations.json", explanations)
        (bundle / "evaluation.md").write_text(render_final_report(summary, metrics, slices), encoding="utf-8")
        plot_final_results(metrics, predicted, predictor, bundle / "holdout_results.png")
        write_json(bundle / "run_manifest.json", {
            "schema_version": 1, "freeze_sha256": sha256(freeze_dir / "freeze.json"),
            "model_sha256": frozen["model_sha256"], "dataset_manifest_sha256": frozen["dataset_manifest_sha256"],
            "split_manifest_sha256": frozen["split_manifest_sha256"], "holdout_ids": ids,
            "protocol": frozen["protocol"], "executable_sha256": frozen["executable_sha256"],
            "files_sha256": {path.name: sha256(path) for path in sorted(bundle.iterdir())},
            "holdout_scored": True, "model_refitted": False,
        })
        if output_dir.exists():
            raise FileExistsError(f"Output appeared during final evaluation: {output_dir}")
        bundle.rename(output_dir)
    return summary
