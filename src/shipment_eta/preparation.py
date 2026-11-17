"""Freeze development splits and evaluate rules on validation windows only."""

from importlib.metadata import version
from pathlib import Path
import json
import platform
import tempfile

import numpy as np
import pandas as pd

from .config import validate_config
from .dataset_io import read_table, sha256
from .evaluation import BASELINES, baseline_prediction, prediction_errors, summarize_errors
from .features import FEATURE_DEFINITIONS, FEATURE_SETS, build_features
from .schema import TARGET_COLUMNS, checkpoint_inputs, validate_inputs, validate_journeys
from .splitting import build_split_plan, load_evaluation_config

DATA_FILES = {"journeys.csv", "checkpoint_inputs.csv", "targets.csv", "audit.json", "simulation_config.json"}


def load_verified_dataset(directory: str | Path):
    """Verify the existing bundle; never regenerate it or silently repair it."""
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or set(manifest.get("files_sha256", {})) != DATA_FILES:
        raise ValueError("Unexpected dataset manifest schema or file list")
    for name in sorted(DATA_FILES):
        if sha256(directory / name) != manifest["files_sha256"][name]:
            raise ValueError(f"Dataset checksum mismatch: {name}")
    config = validate_config(json.loads((directory / "simulation_config.json").read_text(encoding="utf-8")))
    journeys = read_table(directory / "journeys.csv")
    inputs = read_table(directory / "checkpoint_inputs.csv")
    targets = read_table(directory / "targets.csv")
    validate_journeys(journeys, config)
    validate_inputs(inputs)
    try:
        pd.testing.assert_frame_equal(checkpoint_inputs(journeys), inputs, check_dtype=False,
                                      check_exact=False, rtol=0, atol=1e-9)
        pd.testing.assert_frame_equal(journeys[TARGET_COLUMNS], targets, check_dtype=False,
                                      check_exact=False, rtol=0, atol=1e-9)
    except AssertionError as exc:
        raise ValueError("Input/target CSVs disagree with the complete journey table") from exc
    return inputs, targets, {"dataset_manifest_sha256": sha256(manifest_path),
                             "dataset_files_sha256": dict(manifest["files_sha256"])}


def _select(frame, ids):
    return frame.set_index("shipment_id").loc[ids].reset_index()


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def prepare_development(dataset_dir, evaluation_config, output_dir):
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output already exists: {output_dir}. Preserve the frozen split; choose a new directory for an intentional repeat.")
    config = load_evaluation_config(evaluation_config)
    inputs, targets, binding = load_verified_dataset(dataset_dir)
    plan = build_split_plan(inputs, targets, config)
    plan.update(binding)
    plan["evaluation_config_sha256"] = sha256(Path(evaluation_config))
    plan["feature_sets"] = FEATURE_SETS
    development = _select(inputs, plan["development_ids"])
    feature_tables = {name: build_features(development, name) for name in FEATURE_SETS}

    # Diagnostic statistics are restricted to development inputs. No scaling,
    # imputation or model fitting is performed here.
    feature_diagnostics = {}
    for name, features in feature_tables.items():
        matrix = features.to_numpy()
        feature_diagnostics[name] = {
            "rows": len(features), "features": len(features.columns),
            "rank_with_intercept": int(np.linalg.matrix_rank(np.column_stack([np.ones(len(features)), matrix]))),
            "columns_with_intercept": len(features.columns) + 1,
            "constant_columns": [col for col in features if features[col].nunique() < 2],
        }

    metric_rows, records = [], []
    for fold in plan["folds"]:
        validation_inputs = _select(inputs, fold["validation_ids"])
        validation_targets = _select(targets, fold["validation_ids"])
        for baseline in BASELINES:
            prediction = baseline_prediction(validation_inputs, baseline)
            errors = prediction_errors(validation_inputs, validation_targets, prediction)
            metric_rows.append({"baseline": baseline, "scope": f"fold_{fold['fold']}", **summarize_errors(errors)})
            recorded = prediction.join(errors.drop(columns="was_clipped"))
            recorded["fold"] = fold["fold"]
            recorded["baseline"] = baseline
            records.append(recorded.reset_index())
    predictions = pd.concat(records, ignore_index=True)
    if set(predictions.shipment_id).intersection(plan["holdout_ids"]):
        raise ValueError("Holdout IDs unexpectedly appeared in validation predictions")
    if predictions.duplicated(["shipment_id", "baseline"]).any():
        raise ValueError("A shipment was scored twice for the same baseline")
    for baseline in BASELINES:
        rows = predictions.loc[predictions.baseline == baseline]
        metric_rows.append({"baseline": baseline, "scope": "pooled_validation", **summarize_errors(rows)})
    metrics = pd.DataFrame(metric_rows)
    summary = {
        "development_shipments": len(plan["development_ids"]),
        "reserved_holdout_shipments": len(plan["holdout_ids"]),
        "excluded_unavailable_labels": len(plan["excluded_unavailable_label_ids"]),
        "folds": [{"fold": f["fold"], "training_shipments": len(f["training_ids"]),
                   "validation_shipments": len(f["validation_ids"]),
                   "excluded_unavailable_labels": len(f["excluded_unavailable_label_ids"])} for f in plan["folds"]],
        "feature_diagnostics": feature_diagnostics,
        "pooled_validation_metrics": metrics.loc[metrics.scope == "pooled_validation"].to_dict(orient="records"),
        "holdout_scored": False,
        "learned_model_fitted": False,
        "interpretation": "Synthetic-data validation baseline results, not final holdout performance.",
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".development-build-", dir=output_dir.parent) as temporary:
        bundle = Path(temporary) / "bundle"
        bundle.mkdir()
        _write_json(bundle / "split_manifest.json", plan)
        _write_json(bundle / "feature_schema.json", {"sets": FEATURE_SETS, "definitions": FEATURE_DEFINITIONS})
        _write_json(bundle / "summary.json", summary)
        for name, table in feature_tables.items():
            table.to_csv(bundle / f"development_features_{name}.csv", index=True,
                         float_format="%.17g", lineterminator="\n")
        metrics.to_csv(bundle / "baseline_metrics.csv", index=False, float_format="%.17g", lineterminator="\n")
        exported = predictions.copy()
        exported["predicted_eta_utc"] = exported.predicted_eta_utc.map(lambda t: t.isoformat())
        exported.to_csv(bundle / "validation_predictions.csv", index=False,
                        float_format="%.17g", lineterminator="\n")
        source_root = Path(__file__).resolve().parents[2]
        _write_json(bundle / "run_manifest.json", {
            **binding, "evaluation_config_sha256": plan["evaluation_config_sha256"],
            "python": platform.python_version(), "platform": platform.platform(),
            "packages": {name: version(name) for name in ["numpy", "pandas"]},
            "source_sha256": {p.relative_to(source_root).as_posix(): sha256(p)
                              for p in sorted(Path(__file__).parent.glob("*.py"))},
            "files_sha256": {p.name: sha256(p) for p in sorted(bundle.iterdir())},
            "holdout_scored": False, "learned_model_fitted": False,
        })
        if output_dir.exists():
            raise FileExistsError(f"Output appeared during preparation: {output_dir}")
        bundle.rename(output_dir)
    return summary
