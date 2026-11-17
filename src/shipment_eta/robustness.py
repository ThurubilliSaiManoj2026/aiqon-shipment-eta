"""Review saved validation errors and fixed-model behavior before final evaluation."""

from pathlib import Path
import json
import tempfile

import numpy as np
import pandas as pd

from .dataset_io import read_table, sha256
from .evaluation import BASELINES, prediction_errors, prediction_from_remaining, summarize_errors
from .stress import DIRECTION_TOLERANCE_H, invalid_input_checks, stress_responses
from .training import load_frozen_development, load_trusted_model

STAGE6_FILES = {"model.joblib", "model_metadata.json", "selection.json", "summary.json",
                "training_config.json", "fit_audit.json", "candidate_metrics.csv",
                "candidate_validation_predictions.csv", "coefficients.csv"}
MINIMUM_SLICE_N = 10


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _select(frame, ids):
    return frame.set_index("shipment_id").loc[ids].reset_index()


def validate_saved_predictions(rows, inputs, targets, plan):
    """Recompute served predictions/errors from saved raw validation predictions."""
    ids = [sid for fold in plan["folds"] for sid in fold["validation_ids"]]
    if not rows.shipment_id.is_unique or set(rows.shipment_id) != set(ids):
        raise ValueError("Saved predictions do not match the unique frozen validation IDs")
    rows = rows.set_index("shipment_id").loc[ids].copy()
    expected_fold = [fold["fold"] for fold in plan["folds"] for _ in fold["validation_ids"]]
    if rows.fold.tolist() != expected_fold:
        raise ValueError("Saved validation fold membership differs from the frozen plan")
    x, y = _select(inputs, ids), _select(targets, ids)
    predictions = prediction_from_remaining(x, rows.raw_remaining_h)
    for column in ["predicted_remaining_h", "remaining_adjustment_h",
                   "predicted_signed_deviation_h", "predicted_delay_magnitude_h"]:
        if not np.allclose(rows[column], predictions[column], rtol=0, atol=1e-8):
            raise ValueError(f"Saved prediction arithmetic mismatch: {column}")
    if not pd.api.types.is_bool_dtype(rows.was_clipped) or not rows.was_clipped.equals(predictions.was_clipped):
        raise ValueError("Saved clipping flags do not match raw predictions")
    times = pd.to_datetime(rows.predicted_eta_utc, format="ISO8601", utc=True)
    if ((times - predictions.predicted_eta_utc).abs().dt.total_seconds() > 1e-8 * 3600).any():
        raise ValueError("Saved ETA differs from raw remaining-time prediction")
    errors = prediction_errors(x, y, predictions)
    for column in ["eta_error_h", "delay_error_h"]:
        if not np.allclose(errors[column], rows[column], rtol=0, atol=1e-8):
            raise ValueError(f"Saved error differs from validation actuals: {column}")
    # Preserve the recorded validation errors after independent verification.
    return rows.reset_index()


def add_slice_context(errors, inputs, journeys):
    """Future disruption flags enter only this post-outcome diagnostic table."""
    ids = errors.shipment_id.drop_duplicates().tolist()
    context = _select(inputs, ids)[["shipment_id", "route_id", "port_congestion_index",
                                  "actual_port_arrival_utc", "scheduled_port_arrival_utc"]]
    deviation = (context.actual_port_arrival_utc - context.scheduled_port_arrival_utc).dt.total_seconds() / 3600
    context["port_arrival_deviation_h"] = deviation
    context["congestion_slice"] = np.select([context.port_congestion_index < 0.4,
                                              context.port_congestion_index < 0.7], ["low", "medium"], default="high")
    context["arrival_slice"] = np.select([deviation <= 0, deviation <= 12],
                                         ["early_or_on_time", "late_0_to_12h"], default="late_over_12h")
    future = _select(journeys, ids)[["shipment_id", "port_disruption", "inland_disruption", "actual_delivery_utc"]]
    port, inland = future.port_disruption.astype(bool), future.inland_disruption.astype(bool)
    future["future_disruption_slice"] = np.select([port & inland, port, inland],
                                                   ["both", "port_only", "inland_only"], default="none")
    context = context.merge(future, on="shipment_id", validate="one_to_one")
    return errors.merge(context, on="shipment_id", validate="many_to_one", sort=False)


def slice_metrics(validation):
    dimensions = {"all": ["all"], "route_id": sorted(validation.route_id.unique()),
                  "congestion_slice": ["low", "medium", "high"],
                  "arrival_slice": ["early_or_on_time", "late_0_to_12h", "late_over_12h"],
                  "future_disruption_slice": ["none", "port_only", "inland_only", "both"]}
    records = []
    metric_names = ["eta_mae_h", "eta_rmse_h", "eta_bias_h", "delay_mae_h", "clipped_count"]
    for predictor, group in validation.groupby("predictor", sort=True):
        for dimension, values in dimensions.items():
            for value in values:
                rows = group if dimension == "all" else group.loc[group[dimension] == value]
                score = summarize_errors(rows) if len(rows) else {"n": 0, **{name: None for name in metric_names}}
                absolute = rows.eta_error_h.abs()
                records.append({"predictor": predictor, "dimension": dimension, "slice": value,
                                **score, "small_sample": len(rows) < MINIMUM_SLICE_N,
                                "eta_p95_abs_error_h": float(absolute.quantile(.95)) if len(rows) else None,
                                "eta_max_abs_error_h": float(absolute.max()) if len(rows) else None})
    return pd.DataFrame(records)


def paired_comparison(validation, predictor, baseline):
    left = validation.loc[validation.predictor == predictor].set_index("shipment_id").eta_error_h.abs()
    right = validation.loc[validation.predictor == baseline].set_index("shipment_id").eta_error_h.abs()
    if not left.index.is_unique or not right.index.is_unique or set(left.index) != set(right.index):
        raise ValueError("Paired comparison shipment IDs differ")
    improvement = right.loc[left.index] - left
    return {"predictor": predictor, "baseline": baseline, "n": len(left),
            "better": int((improvement > DIRECTION_TOLERANCE_H).sum()),
            "worse": int((improvement < -DIRECTION_TOLERANCE_H).sum()),
            "tied": int((improvement.abs() <= DIRECTION_TOLERANCE_H).sum()),
            "mean_absolute_error_improvement_h": float(improvement.mean()),
            "median_absolute_error_improvement_h": float(improvement.median())}


def plot_route_errors(slices, predictor, baseline, path):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    figure = Figure(figsize=(9, 5), layout="constrained")
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    routes = sorted(slices.loc[slices.dimension == "route_id", "slice"].unique())
    predictors = list(dict.fromkeys([baseline, predictor]))
    colors = ["#748494", "#20689c"]
    for i, name in enumerate(predictors):
        rows = slices.loc[(slices.predictor == name) & (slices.dimension == "route_id")].set_index("slice").loc[routes]
        locations = np.arange(len(routes)) + (i - (len(predictors) - 1) / 2) * 0.36
        bars = axis.bar(locations, rows.eta_mae_h, width=0.34, label=name, color=colors[i])
        axis.bar_label(bars, labels=[f"{v:.2f}" for v in rows.eta_mae_h], padding=3, fontsize=9)
    counts = slices.loc[(slices.predictor == predictor) & (slices.dimension == "route_id")].set_index("slice").loc[routes, "n"]
    axis.set_xticks(np.arange(len(routes)), [f"{r}\nn={n}" for r, n in zip(routes, counts)])
    axis.set_ylabel("Validation ETA MAE (hours; lower is better)")
    axis.set_ylim(0, axis.get_ylim()[1] * 1.25)
    axis.set_title("Route-level errors on the same saved validation shipments")
    axis.legend(fontsize=8)
    figure.supxlabel("Small synthetic samples; selected validation results, not final holdout performance.", fontsize=9)
    figure.savefig(path, dpi=150)


def render_review(summary, slices, checks):
    selected = slices.loc[slices.predictor == summary["predictor"]]
    lines = ["# Development robustness review", "",
             "This report uses saved out-of-fold validation predictions and separate input-response checks. "
             "The final holdout remains unscored. No model was refitted or changed.", "",
             f"Selected predictor: `{summary['predictor']}`. Validation shipments: {summary['validation_shipments']}.", "",
             "## Selected predictor by slice", "",
             "| Dimension | Slice | n | ETA MAE (h) | Bias (h) | Delay MAE (h) | Small sample |",
             "| --- | --- | ---: | ---: | ---: | ---: | --- |"]
    for row in selected.itertuples():
        numbers = ["n/a" if pd.isna(value) else f"{value:.3f}" for value in [row.eta_mae_h, row.eta_bias_h, row.delay_mae_h]]
        lines.append(f"| {row.dimension} | {row.slice} | {row.n} | {' | '.join(numbers)} | {row.small_sample} |")
    pair = summary["paired_vs_best_baseline"]
    lines.extend(["", "Slices overlap across dimensions. n<10 is a display warning, not a statistical test. "
                  "Future-disruption slices use outcome-only flags for diagnosis; these never enter model inputs.", "",
                  "## Comparison with the better fixed baseline", "",
                  f"Against `{pair['baseline']}`: {pair['better']} smaller absolute ETA errors, {pair['worse']} larger, "
                  f"{pair['tied']} tied. Mean absolute-error improvement: {pair['mean_absolute_error_improvement_h']:.6f} h.", "",
                  "This is a paired descriptive comparison after validation selection, not a confidence interval or a causal effect.", "",
                  "## Controlled input-response checks", "",
                  "| Input change | Setting | Comparisons | Directional decreases | Min change (h) | Outside fitted range |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for row in checks:
        delta = "n/a" if row["minimum_directional_change_h"] is None else f"{row['minimum_directional_change_h']:.6f}"
        lines.append(f"| {row['family']} | {row['setting']:g} | {row['directional_comparisons']} | "
                     f"{row['directional_decreases']} | {delta} | {row['outside_training_range_rows']} |")
    lines.extend(["", "Congestion comparisons use adjacent grid levels and remaining time. Inland-plan comparisons use "
                  "remaining time versus the original input. Later-arrival comparisons use final ETA versus the original input. "
                  "Counts of decreases describe potential model behavior limitations, not code-test failures.", "",
                  "No outcome labels exist for these modified scenarios, so these checks do not measure accuracy. "
                  "Out-of-range flags are limited to individual feature ranges. A finite sweep cannot prove global monotonicity.", "",
                  f"Invalid-input cases rejected: {summary['invalid_inputs_rejected']} of {summary['invalid_input_cases']}.", "",
                  "## Review before final evaluation", "",
                  "Inspect small slices, large errors, response decreases and range flags. Preserve the recorded findings. "
                  "Document any decision to retain or change the model before opening the final holdout. "
                  "This command does not tune the model or automatically certify readiness for deployment.", ""])
    return "\n".join(lines)


def review_development(dataset_dir, stage5_dir, model_dir, output_dir):
    output_dir, model_dir = Path(output_dir).resolve(), Path(model_dir)
    if output_dir.exists():
        raise FileExistsError(f"Output already exists: {output_dir}. Preserve it; choose a new output directory.")
    inputs, targets, plan, baselines, binding = load_frozen_development(dataset_dir, stage5_dir)
    manifest = _read_json(model_dir / "run_manifest.json")
    if set(manifest.get("files_sha256", {})) != STAGE6_FILES:
        raise ValueError("Unexpected stage-6 artifact file list")
    for name in sorted(STAGE6_FILES):
        if sha256(model_dir / name) != manifest["files_sha256"][name]:
            raise ValueError(f"Stage-6 checksum mismatch: {name}")
    for key, value in binding.items():
        if manifest.get(key) != value:
            raise ValueError("Model and development artifacts have different provenance")
    if manifest.get("holdout_scored") is not False:
        raise ValueError("Expected an unscored final holdout")
    model, metadata = load_trusted_model(model_dir)
    if metadata["training_ids"] != plan["development_ids"]:
        raise ValueError("Model training IDs differ from the frozen development partition")
    predictor = metadata["validation_recommended_predictor"]
    saved = pd.read_csv(model_dir / "candidate_validation_predictions.csv")
    fixed = pd.read_csv(Path(stage5_dir) / "validation_predictions.csv")
    tables = []
    for name in dict.fromkeys([predictor, *BASELINES]):
        rows = fixed.loc[fixed.baseline == name].copy() if name in BASELINES else saved.loc[saved.candidate == name].copy()
        rows = validate_saved_predictions(rows, inputs, targets, plan)
        rows = rows.drop(columns=[column for column in ["baseline", "candidate"] if column in rows])
        rows["predictor"] = name
        tables.append(rows)
    validation = pd.concat(tables, ignore_index=True)
    if set(validation.shipment_id).intersection(plan["holdout_ids"]):
        raise ValueError("Holdout IDs unexpectedly entered error review")
    journeys = read_table(Path(dataset_dir) / "journeys.csv")
    validation = add_slice_context(validation, inputs, journeys)
    slices = slice_metrics(validation)
    baseline = baselines.sort_values(["eta_mae_h", "baseline"]).iloc[0].baseline
    paired = paired_comparison(validation, predictor, baseline)
    selected_rows = validation.loc[validation.predictor == predictor].copy()
    selected_rows["absolute_eta_error_h"] = selected_rows.eta_error_h.abs()
    largest = selected_rows.sort_values(["absolute_eta_error_h", "shipment_id"], ascending=[False, True], kind="stable").head(10)
    development = _select(inputs, plan["development_ids"])
    responses, checks = stress_responses(model, metadata, development)
    rejected = invalid_input_checks(model, metadata, development)
    overall = slices.loc[(slices.predictor == predictor) & (slices.dimension == "all")].iloc[0]
    summary = {"schema_version": 1, "predictor": predictor, "validation_shipments": len(selected_rows),
               "development_stress_shipments": len(development), "stress_scenario_rows": len(responses),
               "reserved_holdout_shipments": len(plan["holdout_ids"]), "holdout_scored": False,
               "model_refitted": False, "predictor_changed": False,
               "selected_validation_eta_mae_h": float(overall.eta_mae_h),
               "selected_validation_eta_p95_abs_error_h": float(overall.eta_p95_abs_error_h),
               "selected_validation_eta_max_abs_error_h": float(overall.eta_max_abs_error_h),
               "paired_vs_best_baseline": paired,
               "directional_decreases": {family: sum(row["directional_decreases"] for row in checks if row["family"] == family)
                                         for family in ["congestion", "inland_plan", "later_arrival"]},
               "stress_rows_outside_training_ranges": int(responses.outside_training_range.sum()),
               "invalid_input_cases": len(rejected), "invalid_inputs_rejected": int(rejected.rejected.sum()),
               "interpretation": "Selected validation diagnostics and unlabeled input-response checks; not final holdout performance."}
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".review-build-", dir=output_dir.parent) as temporary:
        bundle = Path(temporary) / "bundle"
        bundle.mkdir()
        for name, frame in {"validation_errors.csv": validation, "slice_metrics.csv": slices,
                            "largest_errors.csv": largest, "stress_responses.csv": responses,
                            "invalid_input_checks.csv": rejected}.items():
            frame.to_csv(bundle / name, index=False, float_format="%.17g", lineterminator="\n")
        _write_json(bundle / "stress_checks.json", checks)
        _write_json(bundle / "summary.json", summary)
        (bundle / "review.md").write_text(render_review(summary, slices, checks), encoding="utf-8")
        plot_route_errors(slices, predictor, baseline, bundle / "route_errors.png")
        source_root = Path(__file__).resolve().parents[2]
        _write_json(bundle / "run_manifest.json", {
            "schema_version": 1, **binding, "stage6_manifest_sha256": sha256(model_dir / "run_manifest.json"),
            "model_sha256": sha256(model_dir / "model.joblib"), "model_environment": metadata["environment"],
            "source_sha256": {p.relative_to(source_root).as_posix(): sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
            "files_sha256": {p.name: sha256(p) for p in sorted(bundle.iterdir())},
            "holdout_scored": False, "model_refitted": False,
        })
        if output_dir.exists():
            raise FileExistsError(f"Output appeared during review: {output_dir}")
        bundle.rename(output_dir)
    return summary
