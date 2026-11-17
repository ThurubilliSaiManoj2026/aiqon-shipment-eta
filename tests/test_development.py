"""Features, time splits, rule baselines, metrics and stage-5 integration."""

from pathlib import Path
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from shipment_eta.config import load_config
from shipment_eta.dataset_io import export_dataset, sha256
from shipment_eta.evaluation import (align_targets, baseline_prediction,
                                    prediction_errors, prediction_from_remaining, summarize_errors)
from shipment_eta.features import FEATURE_SETS, build_features
from shipment_eta.preparation import load_verified_dataset, prepare_development
from shipment_eta.schema import TARGET_COLUMNS, checkpoint_inputs
from shipment_eta.simulation import generate_journeys
from shipment_eta.splitting import build_split_plan, load_evaluation_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def data():
    journeys = generate_journeys(load_config(ROOT / "config/simulation.toml"))
    return checkpoint_inputs(journeys), journeys[TARGET_COLUMNS].copy()


@pytest.fixture(scope="module")
def evaluation_config():
    return load_evaluation_config(ROOT / "config/evaluation.toml")


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    destination = tmp_path_factory.mktemp("dataset") / "generated"
    export_dataset(ROOT / "config/simulation.toml", destination)
    return destination


def example_input(data, arrival_h=6):
    row = data[0].head(1).copy()
    anchor = pd.Timestamp("2025-01-04T00:00:00Z")
    for name, hours in {"scheduled_departure_utc": -72, "scheduled_port_arrival_utc": 0,
                        "scheduled_ready_utc": 12, "scheduled_delivery_utc": 28,
                        "actual_departure_utc": arrival_h - 72,
                        "actual_port_arrival_utc": arrival_h, "as_of_utc": arrival_h}.items():
        row[name] = anchor + pd.Timedelta(hours=hours)
    for name, value in {"planned_voyage_h": 72, "actual_voyage_h": 72,
                        "planned_port_processing_h": 12, "planned_inland_h": 8,
                        "delivery_buffer_h": 8, "dispatch_interval_h": 6,
                        "port_congestion_index": 0.5}.items():
        row[name] = float(value)
    return row


def test_hand_calculated_feature_values(data):
    x = build_features(example_input(data, 7.5))
    assert list(x.columns) == FEATURE_SETS["propagation"]
    assert x.iloc[0].port_arrival_deviation_h == 7.5
    assert x.iloc[0].hours_until_first_dispatch == 4.5
    assert x.iloc[0].congestion_processing_h == 6
    assert x.iloc[0].dispatch_phase_sin == pytest.approx(1)
    assert x.iloc[0].dispatch_phase_cos == pytest.approx(0, abs=1e-12)
    assert x.index[0] == data[0].shipment_id.iloc[0]


def test_feature_phase_wrap_and_initial_dispatch_gate(data):
    first = build_features(example_input(data, 12))
    later = build_features(example_input(data, 18))
    np.testing.assert_allclose(first[["dispatch_phase_sin", "dispatch_phase_cos"]],
                               later[["dispatch_phase_sin", "dispatch_phase_cos"]], atol=1e-12)
    assert first.hours_until_first_dispatch.iloc[0] == 0
    assert later.hours_until_first_dispatch.iloc[0] == 0
    assert build_features(example_input(data, -2)).hours_until_first_dispatch.iloc[0] == 14


def test_features_are_row_local_and_do_not_mutate_inputs(data):
    inputs = data[0]
    saved = inputs.copy(deep=True)
    complete = build_features(inputs)
    subset = inputs.sample(n=5, random_state=0)
    pd.testing.assert_frame_equal(build_features(subset), complete.loc[subset.shipment_id])
    pd.testing.assert_frame_equal(inputs, saved)
    assert not {"shipment_id", "route_id", "actual_delivery_utc", "remaining_h"}.intersection(complete.columns)
    assert len(build_features(inputs, "base").columns) == 3


def test_features_reject_future_columns_and_unknown_set(data):
    with pytest.raises(ValueError, match="Column mismatch"):
        build_features(data[0].assign(actual_port_processing_h=10))
    with pytest.raises(ValueError, match="Unknown feature"):
        build_features(data[0], "mystery")


def test_features_reject_unrepresentable_dispatch_interval(data):
    inputs = example_input(data)
    inputs["dispatch_interval_h"] = 1e-20
    with pytest.raises(ValueError, match="timestamp resolution"):
        build_features(inputs)


@pytest.mark.parametrize("arrival,baseline,eta_offset,delay,clipped", [
    (6, "scheduled", 28, 0, False), (6, "carry_forward", 34, 6, False),
    (36, "scheduled", 36, 8, True), (-3, "carry_forward", 25, 0, False),
])
def test_hand_calculated_baselines(data, arrival, baseline, eta_offset, delay, clipped):
    prediction = baseline_prediction(example_input(data, arrival), baseline).iloc[0]
    assert prediction.predicted_eta_utc == pd.Timestamp("2025-01-04T00:00Z") + pd.Timedelta(hours=eta_offset)
    assert prediction.predicted_delay_magnitude_h == delay
    assert bool(prediction.was_clipped) == clipped


def test_carry_forward_remaining_time_identity(data):
    inputs = data[0]
    prediction = baseline_prediction(inputs, "carry_forward")
    np.testing.assert_allclose(prediction.predicted_remaining_h,
                               inputs.planned_port_processing_h + inputs.planned_inland_h + inputs.delivery_buffer_h,
                               rtol=0, atol=1e-9)


def test_negative_prediction_adjustment_is_explicit(data):
    inputs = example_input(data)
    raw = pd.Series([-5.0], index=pd.Index(inputs.shipment_id, name="shipment_id"))
    prediction = prediction_from_remaining(inputs, raw).iloc[0]
    assert prediction.raw_remaining_h == -5
    assert prediction.predicted_remaining_h == 0
    assert prediction.remaining_adjustment_h == 5
    assert prediction.predicted_eta_utc == inputs.actual_port_arrival_utc.iloc[0]


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf, "5", True])
def test_invalid_raw_predictions_rejected(data, bad_value):
    inputs = example_input(data)
    raw = pd.Series([bad_value], index=pd.Index(inputs.shipment_id, name="shipment_id"))
    with pytest.raises(ValueError):
        prediction_from_remaining(inputs, raw)


def test_prediction_id_order_is_checked(data):
    inputs = data[0].head(2)
    raw = pd.Series([1.0, 2.0], index=inputs.shipment_id.iloc[::-1])
    with pytest.raises(ValueError, match="shipment-ID order"):
        prediction_from_remaining(inputs, raw)


def test_target_alignment_uses_ids_and_rejects_corruption(data):
    inputs, targets = data
    aligned = align_targets(inputs, targets.sample(frac=1, random_state=3))
    pd.testing.assert_frame_equal(aligned, targets.set_index("shipment_id"))
    corrupted = targets.copy()
    corrupted.loc[0, "remaining_h"] += 1
    with pytest.raises(ValueError, match="inconsistent"):
        align_targets(inputs, corrupted)
    with pytest.raises(ValueError, match="IDs differ"):
        align_targets(inputs, targets.iloc[:-1])


def test_metrics_match_hand_calculation():
    errors = pd.DataFrame({"eta_error_h": [-3.0, 4.0], "delay_error_h": [-1.0, 2.0],
                           "was_clipped": [False, True]})
    metrics = summarize_errors(errors)
    assert metrics["eta_mae_h"] == 3.5
    assert metrics["eta_rmse_h"] == pytest.approx(np.sqrt(12.5))
    assert metrics["eta_bias_h"] == 0.5
    assert metrics["delay_mae_h"] == 1.5
    assert metrics["clipped_count"] == 1
    with pytest.raises(ValueError, match="empty"):
        summarize_errors(errors.iloc[:0])


def test_timestamp_errors_and_prediction_consistency(data):
    inputs, targets = data
    prediction = baseline_prediction(inputs, "scheduled")
    errors = prediction_errors(inputs, targets, prediction)
    expected = (prediction.predicted_eta_utc.to_numpy() - targets.actual_delivery_utc.to_numpy())
    expected_hours = np.array([delta.total_seconds() / 3600 for delta in expected])
    np.testing.assert_allclose(errors.eta_error_h, expected_hours, atol=1e-9)
    altered = prediction.copy()
    altered.iloc[0, altered.columns.get_loc("predicted_delay_magnitude_h")] += 2
    with pytest.raises(ValueError, match="inconsistent"):
        prediction_errors(inputs, targets, altered)


def test_chronological_splits_and_labels_available_before_every_cutoff(data, evaluation_config):
    inputs, targets = data
    plan = build_split_plan(inputs, targets, evaluation_config)
    assert len(plan["development_ids"]) == 237
    assert len(plan["holdout_ids"]) == 60
    assert len(plan["excluded_unavailable_label_ids"]) == 3
    roles = [set(plan[k]) for k in ["development_ids", "holdout_ids", "excluded_unavailable_label_ids"]]
    assert set.union(*roles) == set(inputs.shipment_id)
    assert all(not roles[i].intersection(roles[j]) for i in range(3) for j in range(i))
    by_id = inputs.set_index("shipment_id")
    delivery = targets.set_index("shipment_id").actual_delivery_utc
    cutoff = pd.Timestamp(plan["holdout_cutoff_utc"])
    assert (delivery.loc[plan["development_ids"]] < cutoff).all()
    assert (delivery.loc[plan["excluded_unavailable_label_ids"]] >= cutoff).all()
    validation_seen = set()
    prior_training = set()
    for fold in plan["folds"]:
        boundary = pd.Timestamp(fold["cutoff_utc"])
        training, validation = fold["training_ids"], fold["validation_ids"]
        assert len(validation) == 40
        assert prior_training.issubset(training)
        assert not set(training).intersection(validation)
        assert not validation_seen.intersection(validation)
        assert (delivery.loc[training] < boundary).all()
        assert (by_id.loc[training, "as_of_utc"] < boundary).all()
        assert (by_id.loc[validation, "as_of_utc"] >= boundary).all()
        assert not set(training + validation).intersection(plan["holdout_ids"])
        assert (delivery.loc[fold["excluded_unavailable_label_ids"]] >= boundary).all()
        validation_seen.update(validation)
        prior_training = set(training)
    assert len(validation_seen) == 120


def test_split_is_stable_under_input_row_permutation(data, evaluation_config):
    inputs, targets = data
    original = build_split_plan(inputs, targets, evaluation_config)
    shuffled = build_split_plan(inputs.sample(frac=1, random_state=3),
                               targets.sample(frac=1, random_state=4), evaluation_config)
    assert shuffled == original


def test_label_at_exact_cutoff_is_excluded(data, evaluation_config):
    inputs, targets = data
    plan = build_split_plan(inputs, targets, evaluation_config)
    altered = targets.copy()
    target_id = plan["development_ids"][-1]
    cutoff = pd.Timestamp(plan["holdout_cutoff_utc"])
    row_index = altered.index[altered.shipment_id == target_id][0]
    checkpoint = inputs.set_index("shipment_id").loc[target_id]
    altered.loc[row_index, "actual_delivery_utc"] = cutoff
    altered.loc[row_index, "remaining_h"] = (cutoff - checkpoint.actual_port_arrival_utc).value / 3_600_000_000_000
    signed = (cutoff - checkpoint.scheduled_delivery_utc).value / 3_600_000_000_000
    altered.loc[row_index, "signed_arrival_deviation_h"] = signed
    altered.loc[row_index, "delay_magnitude_h"] = max(0, signed)
    result = build_split_plan(inputs, altered, evaluation_config)
    assert target_id in result["excluded_unavailable_label_ids"]
    assert target_id not in result["development_ids"]


def test_simultaneous_checkpoints_cannot_be_split_across_holdout_boundary(data, evaluation_config):
    inputs, targets = (frame.copy(deep=True) for frame in data)
    ordered = inputs.sort_values(["as_of_utc", "shipment_id"])
    moved_index = ordered.index[-61]
    cutoff = ordered.iloc[-60].as_of_utc
    target_id = inputs.loc[moved_index, "shipment_id"]
    inputs.loc[moved_index, "actual_port_arrival_utc"] = cutoff
    inputs.loc[moved_index, "as_of_utc"] = cutoff
    inputs.loc[moved_index, "actual_voyage_h"] = (cutoff - inputs.loc[moved_index, "actual_departure_utc"]).value / 3_600_000_000_000
    target_index = targets.index[targets.shipment_id == target_id][0]
    targets.loc[target_index, "remaining_h"] = (targets.loc[target_index, "actual_delivery_utc"] - cutoff).value / 3_600_000_000_000
    with pytest.raises(ValueError, match="simultaneous checkpoints"):
        build_split_plan(inputs, targets, evaluation_config)


@pytest.mark.parametrize("key,value", [("holdout_size", 300), ("validation_size", 100),
                                      ("minimum_training_size", 500), ("n_validation_folds", True)])
def test_infeasible_or_invalid_split_configuration(data, evaluation_config, key, value):
    config = dict(evaluation_config)
    config[key] = value
    with pytest.raises(ValueError):
        build_split_plan(*data, config)


def test_changed_csv_is_detected_before_evaluation(bundle, tmp_path):
    import shutil
    copied = tmp_path / "changed"
    shutil.copytree(bundle, copied)
    path = copied / "checkpoint_inputs.csv"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_verified_dataset(copied)


def test_stage5_outputs_preserve_holdout_and_dataset(bundle, tmp_path):
    original_hashes = {p.name: sha256(p) for p in bundle.iterdir()}
    output = tmp_path / "stage5"
    summary = prepare_development(bundle, ROOT / "config/evaluation.toml", output)
    assert summary["holdout_scored"] is False
    assert summary["learned_model_fitted"] is False
    assert summary["development_shipments"] == 237
    plan = json.loads((output / "split_manifest.json").read_text())
    predictions = pd.read_csv(output / "validation_predictions.csv")
    assert len(predictions) == 240  # 120 unique validation records x 2 baselines
    assert not set(predictions.shipment_id).intersection(plan["holdout_ids"])
    assert set(predictions.shipment_id).issubset(plan["development_ids"])
    for feature_set in FEATURE_SETS:
        frame = pd.read_csv(output / f"development_features_{feature_set}.csv")
        assert len(frame) == 237
        assert set(frame.shipment_id) == set(plan["development_ids"])
    for row in summary["pooled_validation_metrics"]:
        observed = predictions.loc[predictions.baseline == row["baseline"]]
        assert row["eta_mae_h"] == pytest.approx(observed.eta_error_h.abs().mean())
        assert row["n"] == 120
    assert original_hashes == {p.name: sha256(p) for p in bundle.iterdir()}
    manifest = json.loads((output / "run_manifest.json").read_text())
    for name, checksum in manifest["files_sha256"].items():
        assert sha256(output / name) == checksum
    with pytest.raises(FileExistsError):
        prepare_development(bundle, ROOT / "config/evaluation.toml", output)
    repeated = tmp_path / "repeat"
    prepare_development(bundle, ROOT / "config/evaluation.toml", repeated)
    for name in ["split_manifest.json", "baseline_metrics.csv", "validation_predictions.csv", "summary.json"]:
        assert (output / name).read_bytes() == (repeated / name).read_bytes()


def test_cli_works_from_another_directory_and_fails_on_repeat(bundle, tmp_path):
    command = [sys.executable, str(ROOT / "scripts/prepare_development.py"),
               "--data-dir", str(bundle), "--output-dir", str(tmp_path / "output")]
    result = subprocess.run(command, cwd=tmp_path, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert '"holdout_scored": false' in result.stdout
    repeated = subprocess.run(command, cwd=tmp_path, text=True, capture_output=True)
    assert repeated.returncode == 1
    assert "Output already exists" in repeated.stderr
