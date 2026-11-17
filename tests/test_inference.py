"""Independent explanation arithmetic and prediction workflow checks."""

from copy import deepcopy
from pathlib import Path
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from shipment_eta.dataset_io import export_dataset, read_table, sha256, write_table
from shipment_eta.evaluation import baseline_prediction
from shipment_eta.explanations import predict_and_explain
from shipment_eta.features import FEATURE_SETS
from shipment_eta.inference import read_prediction_inputs, run_predictions
from shipment_eta.modeling import fit_candidate, predict_candidate
from shipment_eta.preparation import prepare_development
from shipment_eta.training import load_trusted_model, train_comparison

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def workflow(tmp_path_factory):
    root = tmp_path_factory.mktemp("inference")
    data, stage5, stage6 = root / "data", root / "stage5", root / "stage6"
    export_dataset(ROOT / "config/simulation.toml", data)
    prepare_development(data, ROOT / "config/evaluation.toml", stage5)
    train_comparison(data, stage5, ROOT / "config/training.toml", stage6)
    return data, stage5 / "split_manifest.json", stage6


@pytest.fixture(scope="module")
def loaded(workflow):
    model, metadata = load_trusted_model(workflow[2])
    full = read_prediction_inputs(workflow[0] / "checkpoint_inputs.csv")
    inputs = full.set_index("shipment_id").loc[metadata["training_ids"][:6]].reset_index()
    return model, metadata, inputs


@pytest.fixture(scope="module")
def demonstrated(workflow, tmp_path_factory):
    output = tmp_path_factory.mktemp("prediction-demo") / "bundle"
    summary = run_predictions(workflow[2], workflow[1], output, demo_data_dir=workflow[0])
    return output, summary


def example_input(source, arrival=6.0):
    row = source.head(1).copy()
    anchor = pd.Timestamp("2025-01-04T00:00Z")
    for name, hours in {"scheduled_departure_utc": -72, "scheduled_port_arrival_utc": 0,
                        "scheduled_ready_utc": 12, "scheduled_delivery_utc": 30,
                        "actual_departure_utc": arrival - 72, "actual_port_arrival_utc": arrival,
                        "as_of_utc": arrival}.items():
        row[name] = anchor + pd.Timedelta(hours=hours)
    for name, value in {"planned_voyage_h": 72, "actual_voyage_h": 72,
                        "planned_port_processing_h": 12, "planned_inland_h": 10,
                        "delivery_buffer_h": 8, "dispatch_interval_h": 6,
                        "port_congestion_index": 0.75}.items():
        row[name] = float(value)
    return row


@pytest.fixture(scope="module")
def controlled_model(workflow, loaded):
    inputs = loaded[2]
    targets = read_table(workflow[0] / "targets.csv").set_index("shipment_id").loc[inputs.shipment_id].reset_index()
    model = fit_candidate(inputs, targets, "base", 1.0)
    scaler, ridge = model.named_steps["scale"], model.named_steps["regressor"]
    # Known fitted parameters make the expected arithmetic independent of
    # the actual training result. This controlled object is not persisted.
    scaler.mean_ = np.array([10.0, 20.0, 0.5])
    scaler.scale_ = np.array([2.0, 5.0, 0.25])
    scaler.var_ = scaler.scale_ ** 2
    ridge.coef_ = np.array([3.0, 2.0, 4.0])
    ridge.intercept_ = 30.0
    metadata = {"validation_recommended_predictor": "controlled_ridge",
                "selection": {"candidate": "controlled_ridge", "feature_set": "base"},
                "feature_columns": FEATURE_SETS["base"],
                "training_feature_ranges": {name: {"min": 0.0, "max": 100.0} for name in FEATURE_SETS["base"]}}
    return model, metadata


def test_hand_calculated_contributions_eta_and_delay(controlled_model, loaded):
    inputs = example_input(loaded[2])
    predictions, document = predict_and_explain(*controlled_model, inputs)
    row = document["records"][0]
    assert [t["contribution_h"] for t in row["feature_terms"]] == [3.0, -4.0, 4.0]
    assert row["reference_remaining_h"] == 30
    assert row["raw_remaining_h"] == row["predicted_remaining_h"] == 33
    assert row["predicted_eta_utc"] == "2025-01-05T15:00:00+00:00"
    assert row["deadline_offset_h"] == -24
    assert row["predicted_signed_deviation_h"] == row["predicted_delay_magnitude_h"] == 9
    groups = {g["group"]: g["contribution_h"] for g in row["group_contributions"]}
    assert groups == {"port_processing": 7.0, "inland_plan": -4.0}
    assert row["reconstruction_abs_error_h"] == 0
    assert predictions.predicted_remaining_h.iloc[0] == 33


@pytest.mark.parametrize("arrival,raw,remaining,signed,delay", [
    (6, -10, 0, -24, 0), (36, -10, 0, 6, 6), (6, 0, 0, -24, 0),
    (6, 24, 24, 0, 0), (6, 30, 30, 6, 6),
])
def test_duration_and_delay_clipping_remain_separate(controlled_model, loaded, arrival, raw, remaining, signed, delay):
    model, metadata = deepcopy(controlled_model)
    model.named_steps["regressor"].coef_[:] = 0
    model.named_steps["regressor"].intercept_ = float(raw)
    _, document = predict_and_explain(model, metadata, example_input(loaded[2], arrival))
    record = document["records"][0]
    assert record["raw_remaining_h"] == raw
    assert record["remaining_adjustment_h"] == max(0, -raw)
    assert record["predicted_remaining_h"] == remaining
    assert record["predicted_signed_deviation_h"] == signed
    assert record["predicted_delay_magnitude_h"] == delay
    assert record["delay_floor_adjustment_h"] == max(0, -signed)


def test_every_term_and_group_reconstructs_saved_model_and_preserves_order(loaded):
    model, metadata, inputs = loaded
    inputs = inputs.sample(frac=1, random_state=13)
    before = inputs.copy(deep=True)
    expected = predict_candidate(model, inputs)
    predictions, document = predict_and_explain(model, metadata, inputs)
    pd.testing.assert_frame_equal(predictions, expected)
    pd.testing.assert_frame_equal(inputs, before)
    assert [r["shipment_id"] for r in document["records"]] == inputs.shipment_id.tolist()
    for record in document["records"]:
        terms = record["feature_terms"]
        assert len(terms) == 9
        for term in terms:
            assert term["contribution_h"] == pytest.approx(
                (term["value"] - term["reference_value"]) / term["scale"] * term["coefficient_standardized_h"])
        contribution_sum = sum(t["contribution_h"] for t in terms)
        assert record["raw_remaining_h"] == pytest.approx(record["reference_remaining_h"] + contribution_sum)
        assert sum(g["contribution_h"] for g in record["group_contributions"]) == pytest.approx(contribution_sum)
        expected_eta = pd.Timestamp(record["as_of_utc"]) + pd.Timedelta(hours=record["predicted_remaining_h"])
        assert abs((pd.Timestamp(record["predicted_eta_utc"]) - expected_eta).value) <= 10


@pytest.mark.parametrize("predictor", ["scheduled", "carry_forward"])
def test_saved_baseline_recommendation_is_respected_and_explained(loaded, predictor):
    model, metadata, inputs = loaded
    metadata = deepcopy(metadata)
    metadata["validation_recommended_predictor"] = predictor
    predictions, document = predict_and_explain(model, metadata, inputs)
    pd.testing.assert_frame_equal(predictions, baseline_prediction(inputs, predictor))
    assert document["method"] == "exact_baseline_identity"
    assert document["range_check_applicable"] is False
    for record in document["records"]:
        assert record["reference_remaining_h"] == 0
        assert record["raw_remaining_h"] == pytest.approx(sum(t["contribution_h"] for t in record["feature_terms"]))


def test_feature_range_flags_do_not_modify_predictions(loaded):
    model, metadata, source = loaded
    inputs = source.head(1).copy()
    inputs["planned_inland_h"] = inputs.planned_inland_h + 200
    inputs["scheduled_delivery_utc"] = inputs.scheduled_delivery_utc + pd.Timedelta(hours=200)
    predictions, document = predict_and_explain(model, metadata, inputs)
    pd.testing.assert_frame_equal(predictions, predict_candidate(model, inputs))
    flags = document["records"][0]["training_range_flags"]
    assert "planned_inland_h" in {flag["feature"] for flag in flags}


@pytest.mark.parametrize("change", ["recommendation", "feature_order", "invalid_range", "zero_scale"])
def test_inconsistent_explanation_metadata_is_rejected(loaded, change):
    model, metadata, inputs = deepcopy(loaded)
    if change == "recommendation":
        metadata["validation_recommended_predictor"] = "unknown"
    elif change == "feature_order":
        metadata["feature_columns"] = metadata["feature_columns"][::-1]
    elif change == "invalid_range":
        metadata["training_feature_ranges"]["planned_inland_h"]["min"] = float("nan")
    else:
        model.named_steps["scale"].scale_[0] = 0
    with pytest.raises(ValueError):
        predict_and_explain(model, metadata, inputs)


@pytest.mark.parametrize("change", ["future_column", "duplicate_header", "missing_column", "missing_value",
                                    "duplicate_id", "naive_timestamp", "empty"])
def test_invalid_prediction_csv_rejected(loaded, tmp_path, change):
    inputs = loaded[2].copy()
    path = tmp_path / "input.csv"
    if change == "future_column":
        inputs["actual_delivery_utc"] = inputs.as_of_utc + pd.Timedelta(days=1)
    elif change == "missing_column":
        inputs = inputs.drop(columns="port_congestion_index")
    elif change == "missing_value":
        inputs.loc[0, "port_congestion_index"] = np.nan
    elif change == "duplicate_id":
        inputs.loc[1, "shipment_id"] = inputs.shipment_id.iloc[0]
    elif change == "empty":
        inputs = inputs.iloc[:0]
    write_table(inputs, path)
    if change == "duplicate_header":
        text = path.read_text()
        path.write_text(text.replace("route_id", "shipment_id", 1), encoding="utf-8")
    elif change == "naive_timestamp":
        path.write_text(path.read_text().replace("+00:00", ""), encoding="utf-8")
    with pytest.raises(ValueError):
        read_prediction_inputs(path)


def test_demo_export_has_no_targets_holdout_predictions_or_performance_claims(demonstrated, loaded, workflow):
    output, summary = demonstrated
    assert summary["shipment_ids"] == loaded[1]["training_ids"][:5]
    assert summary["n_predictions"] == 5
    assert summary["holdout_scored"] is summary["performance_evaluated"] is summary["model_refitted"] is False
    assert summary["holdout_ids_predicted"] == 0
    holdout = set(read_json(workflow[1])["holdout_ids"])
    assert not set(summary["shipment_ids"]).intersection(holdout)
    inputs = read_prediction_inputs(output / "checkpoint_inputs.csv")
    assert "actual_delivery_utc" not in inputs
    document = read_json(output / "explanations.json")
    terms = pd.read_csv(output / "contributions.csv")
    assert len(terms) == 45
    for record in document["records"]:
        exported = terms.loc[terms.shipment_id == record["shipment_id"]]
        assert record["raw_remaining_h"] == pytest.approx(record["reference_remaining_h"] + exported.contribution_h.sum())
    for name, checksum in read_json(output / "run_manifest.json")["files_sha256"].items():
        assert sha256(output / name) == checksum
    assert (output / "explanation_example.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    markdown = (output / "explanations.md").read_text(encoding="utf-8")
    assert "not causal effects" in markdown
    assert "no accuracy is measured" in markdown
    assert "delay magnitude" in markdown


def test_checkpoint_only_inference_needs_no_outcome_files_and_changes_no_fitted_state(workflow, loaded, demonstrated, tmp_path, monkeypatch):
    from shipment_eta import training
    data, split, model_dir = workflow
    before = {str(p): sha256(p) for directory in [data, split.parent, model_dir] for p in directory.iterdir()}

    def no_fit(*args, **kwargs):
        pytest.fail("Inference attempted to fit a model")

    monkeypatch.setattr(training, "fit_candidate", no_fit)
    checkpoint_only = tmp_path / "checkpoint_only"
    checkpoint_only.mkdir()
    (checkpoint_only / "checkpoint_inputs.csv").write_bytes((data / "checkpoint_inputs.csv").read_bytes())
    assert not (checkpoint_only / "targets.csv").exists()
    output = tmp_path / "demo"
    run_predictions(model_dir, split, output, demo_data_dir=checkpoint_only)
    custom_output = tmp_path / "csv_mode"
    summary = run_predictions(model_dir, split, custom_output, input_path=output / "checkpoint_inputs.csv")
    assert summary["mode"] == "checkpoint_csv"
    for name in ["predictions.csv", "contributions.csv", "explanations.json"]:
        assert (output / name).read_bytes() == (custom_output / name).read_bytes()
        assert (output / name).read_bytes() == (demonstrated[0] / name).read_bytes()
    assert before == {str(p): sha256(p) for directory in [data, split.parent, model_dir] for p in directory.iterdir()}
    with pytest.raises(FileExistsError):
        run_predictions(model_dir, split, output, demo_data_dir=checkpoint_only)


def test_holdout_guard_rejects_before_prediction(workflow, tmp_path, monkeypatch):
    from shipment_eta import inference

    def no_prediction(*args, **kwargs):
        pytest.fail("Holdout reached the prediction function")

    monkeypatch.setattr(inference, "predict_and_explain", no_prediction)
    with pytest.raises(ValueError, match="Reserved holdout"):
        run_predictions(workflow[2], workflow[1], tmp_path / "rejected",
                        input_path=workflow[0] / "checkpoint_inputs.csv")
    assert not (tmp_path / "rejected").exists()


def test_changed_split_or_demo_input_rejected(workflow, tmp_path):
    split = tmp_path / "split.json"
    split.write_bytes(workflow[1].read_bytes() + b"\n")
    with pytest.raises(ValueError, match="Split manifest"):
        run_predictions(workflow[2], split, tmp_path / "bad_split", demo_data_dir=workflow[0])
    changed = tmp_path / "changed"
    changed.mkdir()
    (changed / "checkpoint_inputs.csv").write_bytes((workflow[0] / "checkpoint_inputs.csv").read_bytes() + b"\n")
    with pytest.raises(ValueError, match="Demo checkpoint"):
        run_predictions(workflow[2], workflow[1], tmp_path / "bad_data", demo_data_dir=changed)


def test_cli_demo_from_another_directory_and_repeat_failure(workflow, tmp_path):
    command = [sys.executable, "-W", "error", str(ROOT / "scripts/predict.py"), "--demo",
               "--model-dir", str(workflow[2]), "--split-manifest", str(workflow[1]),
               "--data-dir", str(workflow[0]), "--output-dir", str(tmp_path / "result")]
    result = subprocess.run(command, cwd=tmp_path, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert '"holdout_scored": false' in result.stdout
    assert '"model_refitted": false' in result.stdout
    repeated = subprocess.run(command, cwd=tmp_path, text=True, capture_output=True)
    assert repeated.returncode == 1
    assert "Output already exists" in repeated.stderr


def test_cli_requires_one_input_mode(tmp_path):
    script = str(ROOT / "scripts/predict.py")
    for args in [[], ["--demo", "--input", "unused.csv"]]:
        result = subprocess.run([sys.executable, script, *args], cwd=tmp_path, capture_output=True, text=True)
        assert result.returncode == 2
