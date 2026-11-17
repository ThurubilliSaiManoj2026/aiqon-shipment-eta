"""Training isolation, frozen artifacts, model selection and persistence."""

from pathlib import Path
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from shipment_eta.dataset_io import export_dataset, sha256
from shipment_eta.features import FEATURE_SETS, build_features
from shipment_eta.modeling import (candidate_id, coefficient_table, fit_candidate,
                                   load_training_config, predict_candidate,
                                   select_candidate, validate_training_config)
from shipment_eta.preparation import load_verified_dataset, prepare_development
from shipment_eta.training import load_frozen_development, load_trusted_model, train_comparison

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def select(frame, ids):
    return frame.set_index("shipment_id").loc[ids].reset_index()


@pytest.fixture(scope="module")
def development(tmp_path_factory):
    root = tmp_path_factory.mktemp("model-development")
    dataset, stage5 = root / "data", root / "stage5"
    export_dataset(ROOT / "config/simulation.toml", dataset)
    prepare_development(dataset, ROOT / "config/evaluation.toml", stage5)
    inputs, targets, _ = load_verified_dataset(dataset)
    return dataset, stage5, inputs, targets, read_json(stage5 / "split_manifest.json")


@pytest.fixture(scope="module")
def trained(development, tmp_path_factory):
    output = tmp_path_factory.mktemp("model-run") / "stage6"
    summary = train_comparison(*development[:2], ROOT / "config/training.toml", output)
    return output, summary


@pytest.fixture(scope="module")
def fold_model(development):
    _, _, inputs, targets, plan = development
    fold = plan["folds"][0]
    train = select(inputs, fold["training_ids"])
    actuals = select(targets, fold["training_ids"])
    return fit_candidate(train, actuals, "propagation", 10.0), train, actuals


@pytest.mark.parametrize("key,value", [
    ("schema_version", True), ("feature_sets", []), ("feature_sets", ["future"]),
    ("feature_sets", [["base"]]), ("feature_sets", ["base", "base"]),
    ("ridge_alphas", []), ("ridge_alphas", [True]), ("ridge_alphas", [0]),
    ("ridge_alphas", [-1]), ("ridge_alphas", [np.nan]), ("ridge_alphas", [np.inf]),
    ("ridge_alphas", [1, 1.0]), ("ridge_solver", "auto"),
    ("selection_mae_tolerance_h", -1), ("selection_mae_tolerance_h", True),
])
def test_invalid_model_config_rejected(key, value):
    config = load_training_config(ROOT / "config/training.toml")
    config[key] = value
    with pytest.raises(ValueError):
        validate_training_config(config)


def test_unknown_settings_rejected_and_candidate_ids_preserve_precision():
    config = load_training_config(ROOT / "config/training.toml")
    with pytest.raises(ValueError, match="keys"):
        validate_training_config({**config, "extra": 1})
    assert candidate_id("base", 1.0000001) != candidate_id("base", 1.0000002)


def test_scaler_uses_training_rows_and_predictions_do_not_refit(fold_model, development):
    model, train, _ = fold_model
    scaler = model.named_steps["scale"]
    features = build_features(train)
    np.testing.assert_allclose(scaler.mean_, features.mean().to_numpy(), atol=1e-12)
    np.testing.assert_allclose(scaler.var_, features.var(ddof=0).to_numpy(), atol=1e-12)
    assert scaler.n_samples_seen_ == len(train) == 114
    saved = (scaler.mean_.copy(), scaler.scale_.copy(), model.named_steps["regressor"].coef_.copy())
    validation = select(development[2], development[4]["folds"][0]["validation_ids"])
    predict_candidate(model, validation)
    for observed, expected in zip((scaler.mean_, scaler.scale_, model.named_steps["regressor"].coef_), saved):
        np.testing.assert_array_equal(observed, expected)
    assert scaler.n_samples_seen_ == 114


def test_ridge_solution_agrees_with_independent_regularized_normal_equations(fold_model):
    model, train, targets = fold_model
    features = build_features(train).to_numpy()
    standardized = (features - features.mean(axis=0)) / features.std(axis=0)
    y = targets.set_index("shipment_id").loc[train.shipment_id, "remaining_h"].to_numpy()
    coefficients = np.linalg.solve(standardized.T @ standardized + 10.0 * np.eye(standardized.shape[1]),
                                   standardized.T @ (y - y.mean()))
    np.testing.assert_allclose(model.named_steps["regressor"].coef_, coefficients, atol=1e-10)
    np.testing.assert_allclose(model.predict(train), y.mean() + standardized @ coefficients, atol=1e-10)


def test_coefficient_export_reconstructs_raw_predictions_and_row_order(fold_model):
    model, train, _ = fold_model
    table = coefficient_table(model, "propagation")
    shuffled = train.sample(n=8, random_state=7)
    values = build_features(shuffled).to_numpy()
    reconstructed = model.named_steps["regressor"].intercept_ + (
        (values - table.training_mean.to_numpy()) * table.coefficient_per_raw_unit_h.to_numpy()).sum(axis=1)
    predictions = predict_candidate(model, shuffled)
    np.testing.assert_allclose(predictions.raw_remaining_h, reconstructed, atol=1e-10)
    pd.testing.assert_frame_equal(predictions, predict_candidate(model, train).loc[shuffled.shipment_id])


def test_shuffled_target_rows_produce_the_same_fit(fold_model):
    model, train, targets = fold_model
    repeated = fit_candidate(train, targets.sample(frac=1, random_state=8), "propagation", 10.0)
    np.testing.assert_allclose(model.predict(train), repeated.predict(train), rtol=0, atol=1e-10)


def test_pipeline_rejects_future_and_missing_inputs(fold_model):
    model, train, _ = fold_model
    with pytest.raises(ValueError, match="Column mismatch"):
        predict_candidate(model, train.assign(actual_delivery_utc=train.as_of_utc))
    bad = train.head(1).copy()
    bad.loc[:, "port_congestion_index"] = np.nan
    with pytest.raises(ValueError):
        predict_candidate(model, bad)


def comparison():
    return pd.DataFrame([
        {"candidate": "p", "feature_set": "propagation", "alpha": 10.0, "eta_mae_h": 5.0},
        {"candidate": "b1", "feature_set": "base", "alpha": 1.0, "eta_mae_h": 5.02},
        {"candidate": "b100", "feature_set": "base", "alpha": 100.0, "eta_mae_h": 5.04},
    ])


def test_selection_tolerance_prefers_simplicity_then_regularization():
    result = select_candidate(comparison(), 0.05)
    assert result["candidate"] == "b100"
    assert result["gap_to_absolute_minimum_h"] == pytest.approx(0.04)
    assert select_candidate(comparison(), 0.01)["candidate"] == "p"
    assert select_candidate(comparison().iloc[::-1], 0.05) == result


def test_invalid_scores_and_duplicate_candidates_rejected():
    bad = comparison()
    bad.loc[0, "eta_mae_h"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        select_candidate(bad, 0.05)
    with pytest.raises(ValueError):
        select_candidate(pd.concat([comparison(), comparison()]), 0.05)


@pytest.mark.parametrize("corruption", ["file", "dataset_binding", "split_ids", "source", "missing_source"])
def test_frozen_stage5_rejects_changed_evidence(development, tmp_path, corruption):
    stage5 = tmp_path / "stage5"
    shutil.copytree(development[1], stage5)
    manifest_path = stage5 / "run_manifest.json"
    manifest = read_json(manifest_path)
    if corruption == "file":
        path = stage5 / "baseline_metrics.csv"
        path.write_bytes(path.read_bytes() + b"\n")
    elif corruption == "dataset_binding":
        manifest["dataset_manifest_sha256"] = "0" * 64
    elif corruption == "split_ids":
        path = stage5 / "split_manifest.json"
        plan = read_json(path)
        plan["development_ids"][0] = plan["holdout_ids"][0]
        write_json(path, plan)
        manifest["files_sha256"][path.name] = sha256(path)
    elif corruption == "source":
        manifest["source_sha256"]["src/shipment_eta/features.py"] = "0" * 64
    else:
        manifest["source_sha256"] = {}
    write_json(manifest_path, manifest)
    with pytest.raises(ValueError):
        load_frozen_development(development[0], stage5)


def test_saved_model_uses_only_eligible_development_and_preserves_holdout(trained, development):
    output, summary = trained
    _, _, inputs, targets, plan = development
    model, metadata = load_trusted_model(output)
    assert metadata["training_ids"] == plan["development_ids"]
    assert model.named_steps["scale"].n_samples_seen_ == metadata["training_size"] == 237
    assert summary["holdout_scored"] is False
    assert summary["reserved_holdout_shipments"] == 60
    assert summary["candidate_count"] == 8
    assert summary["validation_shipments_per_candidate"] == 120
    assert metadata["holdout_scored"] is False
    holdout = set(plan["holdout_ids"])
    assert not set(metadata["training_ids"]).intersection(holdout)
    delivery = targets.set_index("shipment_id").actual_delivery_utc
    assert (delivery.loc[metadata["training_ids"]] < pd.Timestamp(plan["holdout_cutoff_utc"])).all()
    audit = read_json(output / "fit_audit.json")
    assert len(audit) == 24
    for record in audit:
        fold = plan["folds"][record["fold"] - 1]
        assert record["training_ids"] == fold["training_ids"]
        assert record["validation_ids"] == fold["validation_ids"]
        assert record["scaler_samples_seen"] == len(fold["training_ids"])
        assert not set(record["training_ids"] + record["validation_ids"]).intersection(holdout)
        assert (delivery.loc[record["training_ids"]] < pd.Timestamp(fold["cutoff_utc"])).all()
    predictions = pd.read_csv(output / "candidate_validation_predictions.csv")
    assert len(predictions) == 960
    assert not predictions.duplicated(["candidate", "shipment_id"]).any()
    assert not set(predictions.shipment_id).intersection(holdout)
    expected_ids = set().union(*(set(f["validation_ids"]) for f in plan["folds"]))
    for _, rows in predictions.groupby("candidate"):
        assert set(rows.shipment_id) == expected_ids
    coefficients = pd.read_csv(output / "coefficients.csv")
    assert list(coefficients.feature) == FEATURE_SETS[summary["selected_ridge"]["feature_set"]]


def test_metrics_recompute_from_saved_errors_and_selection_obeys_protocol(trained, development):
    output, summary = trained
    scores = pd.read_csv(output / "candidate_metrics.csv")
    predictions = pd.read_csv(output / "candidate_validation_predictions.csv")
    assert len(scores) == 32
    for row in scores.itertuples():
        errors = predictions.loc[predictions.candidate == row.candidate]
        if row.scope != "pooled_validation":
            errors = errors.loc[errors.fold == int(row.scope.split("_")[-1])]
        assert len(errors) == row.n
        assert row.eta_mae_h == pytest.approx(errors.eta_error_h.abs().mean())
        assert row.eta_rmse_h == pytest.approx(np.sqrt((errors.eta_error_h ** 2).mean()))
        assert row.eta_bias_h == pytest.approx(errors.eta_error_h.mean())
        assert row.delay_mae_h == pytest.approx(errors.delay_error_h.abs().mean())
        assert row.clipped_count == int(errors.was_clipped.sum())
    selected = summary["selected_ridge"]
    assert selected["candidate"] == select_candidate(scores.loc[scores.scope == "pooled_validation"], 0.05)["candidate"]
    baselines = pd.read_csv(development[1] / "baseline_metrics.csv")
    baseline = baselines.loc[baselines.scope == "pooled_validation"].sort_values(["eta_mae_h", "baseline"]).iloc[0]
    improvement = baseline.eta_mae_h - selected["validation_eta_mae_h"]
    assert summary["improvement_over_baseline_h"] == pytest.approx(improvement)
    expected = selected["candidate"] if improvement > 0.05 else baseline.baseline
    assert summary["validation_recommended_predictor"] == expected
    for name, checksum in read_json(output / "run_manifest.json")["files_sha256"].items():
        assert sha256(output / name) == checksum


def test_training_calls_are_isolated_reproducible_and_read_only(trained, development, tmp_path, monkeypatch):
    from shipment_eta import training
    inputs, stage5 = development[:2]
    before = {str(p): sha256(p) for folder in (inputs, stage5) for p in folder.iterdir()}
    recorded = []
    original_fit = training.fit_candidate

    def observed_fit(x, y, feature_set, alpha):
        recorded.append((x.shipment_id.tolist(), y.shipment_id.tolist()))
        return original_fit(x, y, feature_set, alpha)

    monkeypatch.setattr(training, "fit_candidate", observed_fit)
    repeat = tmp_path / "repeated"
    summary = train_comparison(inputs, stage5, ROOT / "config/training.toml", repeat)
    assert len(recorded) == 25  # 24 candidate/fold fits + one final development fit
    expected_train = [fold["training_ids"] for _ in range(8) for fold in development[4]["folds"]]
    expected_train.append(development[4]["development_ids"])
    assert recorded == [(ids, ids) for ids in expected_train]
    assert summary == trained[1]
    for name in ["selection.json", "candidate_metrics.csv", "candidate_validation_predictions.csv", "coefficients.csv"]:
        assert (repeat / name).read_bytes() == (trained[0] / name).read_bytes()
    assert before == {str(p): sha256(p) for folder in (inputs, stage5) for p in folder.iterdir()}
    model_hash = sha256(repeat / "model.joblib")
    with pytest.raises(FileExistsError):
        train_comparison(inputs, stage5, ROOT / "config/training.toml", repeat)
    assert sha256(repeat / "model.joblib") == model_hash


def test_saved_pipeline_loads_in_a_fresh_python_process(trained, development, tmp_path):
    input_path = tmp_path / "inputs.csv"
    sample = select(development[2], development[4]["development_ids"][:4])
    sample.to_csv(input_path, index=False)
    output_path = tmp_path / "predicted.json"
    script = (
        "import json,sys; from pathlib import Path; "
        "from shipment_eta.dataset_io import read_table; "
        "from shipment_eta.training import load_trusted_model; "
        "from shipment_eta.modeling import predict_candidate; "
        "model,_=load_trusted_model(sys.argv[1]); "
        "result=predict_candidate(model,read_table(sys.argv[2])); "
        "Path(sys.argv[3]).write_text(json.dumps(result.raw_remaining_h.tolist()))"
    )
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    result = subprocess.run([sys.executable, "-W", "error", "-c", script, str(trained[0]),
                             str(input_path), str(output_path)], cwd=tmp_path, env=env,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    model, _ = load_trusted_model(trained[0])
    np.testing.assert_allclose(read_json(output_path), predict_candidate(model, sample).raw_remaining_h, atol=1e-10)


@pytest.mark.parametrize("corruption", ["model_bytes", "environment", "inference_source"])
def test_bad_saved_artifacts_rejected_before_deserialization(trained, tmp_path, monkeypatch, corruption):
    from shipment_eta import training
    copied = tmp_path / "model"
    shutil.copytree(trained[0], copied)
    manifest_path = copied / "run_manifest.json"
    manifest = read_json(manifest_path)
    if corruption == "model_bytes":
        path = copied / "model.joblib"
        path.write_bytes(path.read_bytes() + b"changed")
    elif corruption == "environment":
        path = copied / "model_metadata.json"
        metadata = read_json(path)
        metadata["environment"]["packages"]["scikit-learn"] = "incompatible"
        write_json(path, metadata)
        manifest["files_sha256"][path.name] = sha256(path)
    else:
        manifest["source_sha256"]["src/shipment_eta/features.py"] = "0" * 64
    write_json(manifest_path, manifest)

    def must_not_load(*args, **kwargs):
        pytest.fail("Deserialization occurred before compatibility checks")

    monkeypatch.setattr(training.joblib, "load", must_not_load)
    with pytest.raises(ValueError):
        load_trusted_model(copied)


def test_training_cli_from_another_directory_and_repeat_failure(development, tmp_path):
    command = [sys.executable, "-W", "error", str(ROOT / "scripts/train_models.py"),
               "--data-dir", str(development[0]), "--stage5-dir", str(development[1]),
               "--output-dir", str(tmp_path / "output")]
    result = subprocess.run(command, cwd=tmp_path, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert '"holdout_scored": false' in result.stdout
    repeated = subprocess.run(command, cwd=tmp_path, text=True, capture_output=True)
    assert repeated.returncode == 1
    assert "Output already exists" in repeated.stderr
