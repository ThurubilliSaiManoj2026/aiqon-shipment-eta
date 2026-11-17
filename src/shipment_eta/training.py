"""Fit candidates on frozen validation folds, then persist a selected Ridge."""

from pathlib import Path
import json
import platform
import tempfile

import joblib
import numpy as np
import pandas as pd

from .dataset_io import sha256
from .evaluation import BASELINES, prediction_errors, summarize_errors
from .features import FEATURE_SETS, build_features
from .modeling import (candidate_id, coefficient_table, current_model_environment,
                       fit_candidate, load_training_config, predict_candidate, select_candidate)
from .preparation import load_verified_dataset
from .splitting import build_split_plan

STAGE5_FILES = {"split_manifest.json", "feature_schema.json", "summary.json",
                "development_features_base.csv", "development_features_propagation.csv",
                "baseline_metrics.csv", "validation_predictions.csv"}
INFERENCE_SOURCES = ("features.py", "schema.py", "timing.py", "evaluation.py", "modeling.py")
STAGE5_CORE_SOURCES = ("__init__.py", "config.py", "dataset_io.py", "evaluation.py", "features.py",
                       "preparation.py", "schema.py", "simulation.py", "splitting.py", "timing.py")


def _select(frame, ids):
    return frame.set_index("shipment_id").loc[ids].reset_index()


def _json(path, content):
    path.write_text(json.dumps(content, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def load_frozen_development(dataset_dir, stage5_dir):
    inputs, targets, dataset_binding = load_verified_dataset(dataset_dir)
    stage5_dir = Path(stage5_dir)
    run_path = stage5_dir / "run_manifest.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    if set(run.get("files_sha256", {})) != STAGE5_FILES:
        raise ValueError("Unexpected stage-5 artifact file list")
    for name in sorted(STAGE5_FILES):
        if sha256(stage5_dir / name) != run["files_sha256"][name]:
            raise ValueError(f"Stage-5 checksum mismatch: {name}")
    for name, expected in dataset_binding.items():
        if run.get(name) != expected:
            raise ValueError("Stage-5 artifacts refer to a different dataset")
    plan_path = stage5_dir / "split_manifest.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    for name, expected in dataset_binding.items():
        if plan.get(name) != expected:
            raise ValueError("Split manifest refers to a different dataset")
    expected_plan = build_split_plan(inputs, targets, plan["evaluation_config"])
    for name, expected in expected_plan.items():
        if plan.get(name) != expected:
            raise ValueError(f"Frozen split disagrees with its declared policy: {name}")
    if plan.get("feature_sets") != FEATURE_SETS:
        raise ValueError("Feature definitions changed since stage 5")
    if run.get("holdout_scored") is not False or run.get("learned_model_fitted") is not False:
        raise ValueError("Expected unscored stage-5 development artifacts")
    source_root = Path(__file__).resolve().parents[2]
    required_sources = {f"src/shipment_eta/{name}" for name in STAGE5_CORE_SOURCES}
    if not required_sources.issubset(run.get("source_sha256", {})):
        raise ValueError("Incomplete stage-5 source provenance")
    for name, checksum in run.get("source_sha256", {}).items():
        relative = Path(name)
        if relative.parts[:2] != ("src", "shipment_eta") or len(relative.parts) != 3 or relative.suffix != ".py":
            raise ValueError("Unexpected source path in stage-5 manifest")
        if sha256(source_root / relative) != checksum:
            raise ValueError(f"Source changed since stage 5: {name}")
    baseline = pd.read_csv(stage5_dir / "baseline_metrics.csv")
    baseline = baseline.loc[baseline.scope == "pooled_validation"].copy()
    validation_n = sum(len(fold["validation_ids"]) for fold in plan["folds"])
    if len(baseline) != len(BASELINES) or set(baseline.baseline) != set(BASELINES) or not (baseline.n == validation_n).all():
        raise ValueError("Unexpected baseline comparison rows")
    if not np.isfinite(baseline.eta_mae_h).all() or (baseline.eta_mae_h < 0).any():
        raise ValueError("Invalid baseline scores")
    binding = {**dataset_binding, "stage5_manifest_sha256": sha256(run_path),
               "split_manifest_sha256": sha256(plan_path)}
    return inputs, targets, plan, baseline, binding


def train_comparison(dataset_dir, stage5_dir, config_path, output_dir):
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output already exists: {output_dir}. Preserve this run; choose a new directory for an intentional repeat.")
    config = load_training_config(config_path)
    inputs, targets, plan, baselines, binding = load_frozen_development(dataset_dir, stage5_dir)
    metrics, predictions, fit_audit = [], [], []
    for feature_set in config["feature_sets"]:
        for alpha in config["ridge_alphas"]:
            name = candidate_id(feature_set, alpha)
            per_candidate = []
            for fold in plan["folds"]:
                train_inputs = _select(inputs, fold["training_ids"])
                train_targets = _select(targets, fold["training_ids"])
                validation_inputs = _select(inputs, fold["validation_ids"])
                validation_targets = _select(targets, fold["validation_ids"])
                pipeline = fit_candidate(train_inputs, train_targets, feature_set, alpha)
                output = predict_candidate(pipeline, validation_inputs)
                errors = prediction_errors(validation_inputs, validation_targets, output)
                identity = {"candidate": name, "feature_set": feature_set, "alpha": alpha}
                metrics.append({**identity, "scope": f"fold_{fold['fold']}", **summarize_errors(errors)})
                record = output.join(errors.drop(columns="was_clipped")).reset_index()
                record["candidate"], record["fold"] = name, fold["fold"]
                predictions.append(record)
                per_candidate.append(errors)
                fit_audit.append({**identity, "fold": fold["fold"],
                                  "training_ids": fold["training_ids"],
                                  "validation_ids": fold["validation_ids"],
                                  "scaler_samples_seen": int(pipeline.named_steps["scale"].n_samples_seen_)})
            metrics.append({**identity, "scope": "pooled_validation",
                            **summarize_errors(pd.concat(per_candidate))})
    metric_frame = pd.DataFrame(metrics)
    prediction_frame = pd.concat(predictions, ignore_index=True)
    if prediction_frame.duplicated(["candidate", "shipment_id"]).any():
        raise ValueError("Duplicate validation prediction for a candidate")
    if set(prediction_frame.shipment_id).intersection(plan["holdout_ids"]):
        raise ValueError("Holdout IDs unexpectedly occurred in model predictions")
    pooled = metric_frame.loc[metric_frame.scope == "pooled_validation"].copy()
    selection = select_candidate(pooled, config["selection_mae_tolerance_h"])
    best_baseline = baselines.sort_values(["eta_mae_h", "baseline"]).iloc[0]
    improvement = float(best_baseline.eta_mae_h - selection["validation_eta_mae_h"])
    recommended = selection["candidate"] if improvement > config["selection_mae_tolerance_h"] else best_baseline.baseline

    development_inputs = _select(inputs, plan["development_ids"])
    development_targets = _select(targets, plan["development_ids"])
    final_model = fit_candidate(development_inputs, development_targets,
                                selection["feature_set"], selection["alpha"])
    features = build_features(development_inputs, selection["feature_set"])
    parameters = coefficient_table(final_model, selection["feature_set"])
    environment = current_model_environment()
    metadata = {"schema_version": 1, "model_type": "standardized_ridge_remaining_hours",
                "selection": selection, "environment": environment,
                "python_patch_version": platform.python_version(),
                "training_ids": plan["development_ids"], "training_size": len(development_inputs),
                "feature_columns": FEATURE_SETS[selection["feature_set"]],
                "standardized_intercept_h": float(final_model.named_steps["regressor"].intercept_),
                "training_feature_ranges": {name: {"min": float(features[name].min()), "max": float(features[name].max())}
                                            for name in features},
                "validation_recommended_predictor": recommended,
                "holdout_scored": False, **binding}
    summary = {"candidate_count": len(pooled), "fold_count": len(plan["folds"]),
               "validation_shipments_per_candidate": int(pooled.n.iloc[0]),
               "final_training_shipments": len(development_inputs), "reserved_holdout_shipments": len(plan["holdout_ids"]),
               "selected_ridge": selection,
               "best_fixed_baseline": {"name": best_baseline.baseline, "validation_eta_mae_h": float(best_baseline.eta_mae_h)},
               "improvement_over_baseline_h": improvement,
               "validation_recommended_predictor": recommended,
               "group_best_validation_mae_h": {group: float(rows.eta_mae_h.min()) for group, rows in pooled.groupby("feature_set")},
               "holdout_scored": False, "learned_model_fitted": True,
               "interpretation": "Selected validation scores on synthetic data, not final holdout estimates."}
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".model-build-", dir=output_dir.parent) as temporary:
        bundle = Path(temporary) / "bundle"
        bundle.mkdir()
        joblib.dump(final_model, bundle / "model.joblib")
        # Reload only the artifact just created here, never an untrusted pickle.
        restored = joblib.load(bundle / "model.joblib")
        check_inputs = development_inputs.head(10)
        before = predict_candidate(final_model, check_inputs).raw_remaining_h.to_numpy()
        after = predict_candidate(restored, check_inputs).raw_remaining_h.to_numpy()
        np.testing.assert_allclose(before, after, rtol=0, atol=1e-10)
        summary["serialization_max_abs_difference_h"] = float(np.max(np.abs(before - after)))
        _json(bundle / "model_metadata.json", metadata)
        _json(bundle / "selection.json", selection)
        _json(bundle / "summary.json", summary)
        _json(bundle / "training_config.json", config)
        _json(bundle / "fit_audit.json", fit_audit)
        metric_frame.to_csv(bundle / "candidate_metrics.csv", index=False, float_format="%.17g", lineterminator="\n")
        exported = prediction_frame.copy()
        exported["predicted_eta_utc"] = exported.predicted_eta_utc.map(lambda stamp: stamp.isoformat())
        exported.to_csv(bundle / "candidate_validation_predictions.csv", index=False,
                        float_format="%.17g", lineterminator="\n")
        parameters.to_csv(bundle / "coefficients.csv", index=False, float_format="%.17g", lineterminator="\n")
        source_root = Path(__file__).resolve().parents[2]
        _json(bundle / "run_manifest.json", {
            **binding, "training_config_sha256": sha256(Path(config_path)),
            "environment": environment, "python": platform.python_version(), "platform": platform.platform(),
            "source_sha256": {p.relative_to(source_root).as_posix(): sha256(p)
                              for p in sorted(Path(__file__).parent.glob("*.py"))},
            "files_sha256": {p.name: sha256(p) for p in sorted(bundle.iterdir())},
            "holdout_scored": False, "learned_model_fitted": True,
        })
        if output_dir.exists():
            raise FileExistsError(f"Output appeared during training: {output_dir}")
        bundle.rename(output_dir)
    return summary


def load_trusted_model(directory):
    """Load this project's own artifact after integrity/environment checks.

    Hashes do not make an untrusted joblib/pickle safe. Call only for a model
    produced by this project in a trusted local workflow.
    """
    directory = Path(directory)
    manifest = json.loads((directory / "run_manifest.json").read_text(encoding="utf-8"))
    metadata = json.loads((directory / "model_metadata.json").read_text(encoding="utf-8"))
    for name in ["model.joblib", "model_metadata.json"]:
        if sha256(directory / name) != manifest.get("files_sha256", {}).get(name):
            raise ValueError(f"Model artifact checksum mismatch: {name}")
    if metadata.get("schema_version") != 1 or metadata.get("environment") != current_model_environment():
        raise ValueError("Model environment/schema mismatch; recreate the recorded environment or retrain")
    for name in INFERENCE_SOURCES:
        key = f"src/shipment_eta/{name}"
        if manifest.get("source_sha256", {}).get(key) != sha256(Path(__file__).parent / name):
            raise ValueError(f"Inference source changed since training: {name}")
    model = joblib.load(directory / "model.joblib")
    if list(model.named_steps["scale"].feature_names_in_) != metadata["feature_columns"]:
        raise ValueError("Saved feature schema differs from fitted model")
    return model, metadata
