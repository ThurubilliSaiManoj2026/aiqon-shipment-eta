"""Bind final evaluation to reviewed data, fitted model and executable source."""

from pathlib import Path
import json
import tempfile

from .dataset_io import sha256
from .evaluation import BASELINES
from .robustness import STAGE6_FILES
from .training import load_frozen_development, load_trusted_model

STAGE8_FILES = {"validation_errors.csv", "slice_metrics.csv", "largest_errors.csv",
                "stress_responses.csv", "invalid_input_checks.csv", "stress_checks.json",
                "summary.json", "review.md", "route_errors.png"}
FINAL_PROTOCOL = {
    "schema_version": 1, "primary_metric": "eta_mae_h", "baseline_names": list(BASELINES),
    "additional_metrics": ["eta_rmse_h", "eta_bias_h", "delay_mae_h", "clipped_count",
                           "eta_p95_abs_error_h", "eta_max_abs_error_h"],
    "prediction_rule": "Clip remaining hours at zero; ETA=checkpoint+remaining; delay=max(0,ETA-deadline).",
    "comparison_baseline_selection": "Lowest pooled validation ETA MAE; baseline name breaks ties.",
    "bootstrap_resamples": 5000, "bootstrap_seed": 2026, "bootstrap_generator": "PCG64",
    "bootstrap_confidence_level": 0.95, "bootstrap_method": "paired_shipments_percentile",
    "bootstrap_quantile_method": "linear", "small_slice_n": 10,
    "largest_errors_count": 10, "model_refitting": False, "post_holdout_tuning": False,
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, content):
    path.write_text(json.dumps(content, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def verify_bundle(directory, expected_files):
    directory = Path(directory)
    manifest = read_json(directory / "run_manifest.json")
    if set(manifest.get("files_sha256", {})) != expected_files:
        raise ValueError(f"Unexpected artifact list in {directory.name}")
    for name in sorted(expected_files):
        if sha256(directory / name) != manifest["files_sha256"][name]:
            raise ValueError(f"Artifact checksum mismatch: {directory.name}/{name}")
    return manifest


def executable_snapshot():
    root = Path(__file__).resolve().parents[2]
    paths = list((root / "src/shipment_eta").glob("*.py")) + list((root / "scripts").glob("*.py"))
    paths += list((root / "config").glob("*.toml"))
    paths.append(root / "docs/FINAL_EVALUATION_PROTOCOL.md")
    return {path.relative_to(root).as_posix(): sha256(path) for path in sorted(paths)}


def freeze_context(dataset_dir, stage5_dir, model_dir, review_dir):
    """Validate pre-evaluation evidence without making predictions."""
    inputs, targets, plan, baselines, binding = load_frozen_development(dataset_dir, stage5_dir)
    model_dir, review_dir = Path(model_dir), Path(review_dir)
    model_manifest = verify_bundle(model_dir, STAGE6_FILES)
    review_manifest = verify_bundle(review_dir, STAGE8_FILES)
    for name, expected in binding.items():
        if model_manifest.get(name) != expected or review_manifest.get(name) != expected:
            raise ValueError("Model/review provenance differs from frozen development evidence")
    model_manifest_hash = sha256(model_dir / "run_manifest.json")
    model_hash = sha256(model_dir / "model.joblib")
    if review_manifest.get("stage6_manifest_sha256") != model_manifest_hash or review_manifest.get("model_sha256") != model_hash:
        raise ValueError("Development review refers to a different fitted model")
    if model_manifest.get("holdout_scored") is not False or review_manifest.get("holdout_scored") is not False:
        raise ValueError("Expected pre-evaluation model and review records")
    root = Path(__file__).resolve().parents[2]
    for name, expected in review_manifest.get("source_sha256", {}).items():
        relative = Path(name)
        if relative.parts[:2] != ("src", "shipment_eta") or len(relative.parts) != 3 or relative.suffix != ".py":
            raise ValueError("Unexpected reviewed-source path")
        if sha256(root / relative) != expected:
            raise ValueError(f"Source changed since development review: {name}")
    model, metadata = load_trusted_model(model_dir)
    review = read_json(review_dir / "summary.json")
    predictor = metadata["validation_recommended_predictor"]
    if metadata["training_ids"] != plan["development_ids"]:
        raise ValueError("Fitted model training IDs differ from eligible development IDs")
    if review.get("predictor") != predictor or review.get("model_refitted") is not False or review.get("predictor_changed") is not False:
        raise ValueError("Development review does not describe the unchanged selected predictor")
    if review.get("invalid_input_cases") != 8 or review.get("invalid_inputs_rejected") != 8:
        raise ValueError("Development input-contract checks are incomplete")
    comparison = baselines.sort_values(["eta_mae_h", "baseline"]).iloc[0].baseline
    frozen = {
        "schema_version": 1, "protocol": json.loads(json.dumps(FINAL_PROTOCOL)), **binding,
        "stage6_manifest_sha256": model_manifest_hash, "model_sha256": model_hash,
        "model_metadata_sha256": sha256(model_dir / "model_metadata.json"),
        "stage8_manifest_sha256": sha256(review_dir / "run_manifest.json"),
        "executable_sha256": executable_snapshot(), "model_environment": metadata["environment"],
        "predictor": predictor, "selection": metadata["selection"], "feature_columns": metadata["feature_columns"],
        "comparison_baseline": comparison, "training_ids": plan["development_ids"],
        "holdout_ids": plan["holdout_ids"], "excluded_unavailable_label_ids": plan["excluded_unavailable_label_ids"],
        "holdout_cutoff_utc": plan["holdout_cutoff_utc"], "holdout_scored": False,
    }
    return frozen, inputs, targets, model, metadata


def freeze_evaluation(dataset_dir, stage5_dir, model_dir, review_dir, output_dir):
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output already exists: {output_dir}. Preserve the existing freeze.")
    frozen, _, _, _, _ = freeze_context(dataset_dir, stage5_dir, model_dir, review_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".freeze-build-", dir=output_dir.parent) as temporary:
        bundle = Path(temporary) / "bundle"
        bundle.mkdir()
        write_json(bundle / "freeze.json", frozen)
        write_json(bundle / "run_manifest.json", {"schema_version": 1,
                   "files_sha256": {"freeze.json": sha256(bundle / "freeze.json")}, "holdout_scored": False})
        if output_dir.exists():
            raise FileExistsError(f"Output appeared while freezing: {output_dir}")
        bundle.rename(output_dir)
    return {"predictor": frozen["predictor"], "comparison_baseline": frozen["comparison_baseline"],
            "training_shipments": len(frozen["training_ids"]), "holdout_shipments": len(frozen["holdout_ids"]),
            "excluded_unavailable_labels": len(frozen["excluded_unavailable_label_ids"]),
            "primary_metric": FINAL_PROTOCOL["primary_metric"], "bootstrap_resamples": FINAL_PROTOCOL["bootstrap_resamples"],
            "freeze_sha256": sha256(output_dir / "freeze.json"), "holdout_scored": False}
