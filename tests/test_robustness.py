"""Audit computations, coherent stress inputs and preserved evaluation boundaries."""

from copy import deepcopy
from pathlib import Path
import json
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from shipment_eta.dataset_io import export_dataset, read_table, sha256
from shipment_eta.evaluation import prediction_from_remaining
from shipment_eta.preparation import prepare_development
from shipment_eta.robustness import (add_slice_context, paired_comparison, review_development,
                                     slice_metrics, validate_saved_predictions)
from shipment_eta.schema import validate_inputs
from shipment_eta.stress import invalid_input_checks, perturb, stress_responses
from shipment_eta.training import load_trusted_model, train_comparison

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def evidence(tmp_path_factory):
    root = tmp_path_factory.mktemp("robustness-evidence")
    data, stage5, model = root / "data", root / "stage5", root / "model"
    export_dataset(ROOT / "config/simulation.toml", data)
    prepare_development(data, ROOT / "config/evaluation.toml", stage5)
    train_comparison(data, stage5, ROOT / "config/training.toml", model)
    return data, stage5, model


@pytest.fixture(scope="module")
def loaded(evidence):
    model, metadata = load_trusted_model(evidence[2])
    inputs = read_table(evidence[0] / "checkpoint_inputs.csv")
    inputs = inputs.set_index("shipment_id").loc[metadata["training_ids"]].reset_index()
    return model, metadata, inputs


@pytest.fixture(scope="module")
def reviewed(evidence, tmp_path_factory):
    output = tmp_path_factory.mktemp("review") / "stage8"
    summary = review_development(*evidence, output)
    return output, summary


@pytest.mark.parametrize("family,amount", [("congestion", .75), ("inland_plan", 6.0), ("later_arrival", 6.0)])
def test_perturbations_preserve_input_identities_and_original(loaded, family, amount):
    original = loaded[2].head(3).copy()
    before = original.copy(deep=True)
    altered = perturb(original, family, amount)
    validate_inputs(altered)
    pd.testing.assert_frame_equal(original, before)
    assert altered.shipment_id.tolist() == original.shipment_id.tolist()
    expected_changes = {"congestion": {"port_congestion_index"},
                        "inland_plan": {"planned_inland_h", "scheduled_delivery_utc"},
                        "later_arrival": {"actual_port_arrival_utc", "as_of_utc", "actual_voyage_h"}}[family]
    for column in set(original.columns) - expected_changes:
        pd.testing.assert_series_equal(altered[column], original[column])
    if family == "congestion":
        assert (altered.port_congestion_index == amount).all()
    elif family == "inland_plan":
        np.testing.assert_allclose(altered.planned_inland_h - original.planned_inland_h, amount)
        assert ((altered.scheduled_delivery_utc - original.scheduled_delivery_utc) == pd.Timedelta(hours=amount)).all()
    else:
        np.testing.assert_allclose(altered.actual_voyage_h - original.actual_voyage_h, amount)
        assert ((altered.as_of_utc - original.as_of_utc) == pd.Timedelta(hours=amount)).all()


@pytest.mark.parametrize("family,amount", [("congestion", -0.1), ("congestion", 1.1),
                                          ("inland_plan", 0), ("later_arrival", -1), ("unknown", 1)])
def test_invalid_perturbations_rejected(loaded, family, amount):
    with pytest.raises(ValueError):
        perturb(loaded[2].head(1), family, amount)


def test_carry_forward_stress_changes_match_hand_calculation(loaded):
    model, metadata, inputs = loaded
    metadata = deepcopy(metadata)
    metadata["validation_recommended_predictor"] = "carry_forward"
    responses, checks = stress_responses(model, metadata, inputs)
    assert len(responses) == 237 * 9
    np.testing.assert_allclose(responses.loc[responses.family == "congestion", "remaining_change_h"], 0, atol=1e-9)
    inland = responses.loc[responses.family == "inland_plan"]
    np.testing.assert_allclose(inland.remaining_change_h, 6, atol=1e-9)
    np.testing.assert_allclose(inland.eta_change_h, 6, atol=1e-9)
    np.testing.assert_allclose(inland.delay_change_h, 0, atol=1e-9)
    arrival = responses.loc[responses.family == "later_arrival"]
    np.testing.assert_allclose(arrival.eta_change_h, arrival.setting, atol=1e-9)
    np.testing.assert_allclose(arrival.remaining_change_h, 0, atol=1e-9)
    assert sum(check["directional_decreases"] for check in checks) == 0


def test_auditor_detects_directional_reversals_without_suppressing_them(loaded, monkeypatch):
    from shipment_eta import stress
    model, metadata, inputs = loaded
    metadata = deepcopy(metadata)
    metadata["validation_recommended_predictor"] = "carry_forward"

    def deliberately_wrong_response(model, metadata, frame):
        deviation = (frame.actual_port_arrival_utc - frame.scheduled_port_arrival_utc).dt.total_seconds() / 3600
        raw = 1000 - 10 * frame.port_congestion_index - 2 * frame.planned_inland_h - 2 * deviation
        return prediction_from_remaining(frame, pd.Series(raw.to_numpy(), index=pd.Index(frame.shipment_id, name="shipment_id")))

    monkeypatch.setattr(stress, "serve", deliberately_wrong_response)
    _, checks = stress_responses(model, metadata, inputs)
    totals = {family: sum(row["directional_decreases"] for row in checks if row["family"] == family)
              for family in ["congestion", "inland_plan", "later_arrival"]}
    assert totals == {"congestion": 948, "inland_plan": 237, "later_arrival": 711}


def test_stress_ids_must_be_exactly_saved_development(loaded):
    with pytest.raises(ValueError, match="exactly"):
        stress_responses(loaded[0], loaded[1], loaded[2].iloc[::-1])


def test_invalid_input_audit_fails_if_a_case_is_accepted(loaded, monkeypatch):
    from shipment_eta import stress
    monkeypatch.setattr(stress, "serve", lambda *args: pd.DataFrame())
    with pytest.raises(ValueError, match="unexpectedly accepted"):
        invalid_input_checks(*loaded)


def test_slice_boundaries_and_disruption_flags_are_explicit():
    anchor = pd.Timestamp("2025-01-01T00:00Z")
    inputs = pd.DataFrame({"shipment_id": ["a", "b", "c", "d"], "route_id": ["R"] * 4,
                           "port_congestion_index": [0.0, 0.4, 0.7, 1.0],
                           "scheduled_port_arrival_utc": [anchor] * 4,
                           "actual_port_arrival_utc": [anchor, anchor + pd.Timedelta(hours=12),
                                                       anchor + pd.Timedelta(hours=13), anchor - pd.Timedelta(hours=1)]})
    journeys = pd.DataFrame({"shipment_id": ["a", "b", "c", "d"], "port_disruption": [0, 1, 0, 1],
                             "inland_disruption": [0, 0, 1, 1], "actual_delivery_utc": [anchor + pd.Timedelta(days=2)] * 4})
    result = add_slice_context(pd.DataFrame({"shipment_id": ["a", "b", "c", "d"]}), inputs, journeys)
    assert result.congestion_slice.tolist() == ["low", "medium", "high", "high"]
    assert result.arrival_slice.tolist() == ["early_or_on_time", "late_0_to_12h", "late_over_12h", "early_or_on_time"]
    assert result.future_disruption_slice.tolist() == ["none", "port_only", "inland_only", "both"]


def test_slice_metrics_match_hand_calculation_and_report_empty_groups():
    data = pd.DataFrame({"predictor": ["test"] * 2, "route_id": ["R"] * 2,
                         "congestion_slice": ["low"] * 2, "arrival_slice": ["late_0_to_12h"] * 2,
                         "future_disruption_slice": ["none"] * 2,
                         "eta_error_h": [-3.0, 4.0], "delay_error_h": [-1.0, 2.0], "was_clipped": [False, True]})
    result = slice_metrics(data)
    overall = result.loc[result.dimension == "all"].iloc[0]
    assert overall.eta_mae_h == 3.5
    assert overall.eta_rmse_h == pytest.approx(np.sqrt(12.5))
    assert overall.eta_bias_h == .5
    assert overall.delay_mae_h == 1.5
    assert overall.eta_p95_abs_error_h == pytest.approx(3.95)
    assert overall.eta_max_abs_error_h == 4
    assert bool(overall.small_sample)
    empty = result.loc[(result.dimension == "future_disruption_slice") & (result["slice"] == "both")].iloc[0]
    assert empty.n == 0
    assert pd.isna(empty.eta_mae_h)


def test_paired_comparison_aligns_ids_and_rejects_duplicates():
    frame = pd.DataFrame({"shipment_id": ["a", "b", "c", "c", "a", "b"],
                          "predictor": ["model"] * 3 + ["baseline"] * 3,
                          "eta_error_h": [-1.0, 3.0, 2.0, -2.0, 2.0, -1.0]})
    result = paired_comparison(frame, "model", "baseline")
    assert (result["better"], result["worse"], result["tied"]) == (1, 1, 1)
    assert result["mean_absolute_error_improvement_h"] == pytest.approx(-1 / 3)
    with pytest.raises(ValueError):
        paired_comparison(pd.concat([frame, frame.iloc[[-1]]]), "model", "baseline")


@pytest.mark.parametrize("corruption", ["id", "fold", "remaining", "eta", "error", "clip"])
def test_saved_validation_records_are_independently_checked(evidence, corruption):
    plan = read_json(evidence[1] / "split_manifest.json")
    inputs = read_table(evidence[0] / "checkpoint_inputs.csv")
    targets = read_table(evidence[0] / "targets.csv")
    all_rows = pd.read_csv(evidence[2] / "candidate_validation_predictions.csv")
    selected = read_json(evidence[2] / "selection.json")["candidate"]
    rows = all_rows.loc[all_rows.candidate == selected].copy().reset_index(drop=True)
    if corruption == "id":
        rows.loc[0, "shipment_id"] = plan["holdout_ids"][0]
    elif corruption == "fold":
        rows.loc[0, "fold"] = 2
    elif corruption == "remaining":
        rows.loc[0, "predicted_remaining_h"] += 1
    elif corruption == "eta":
        rows.loc[0, "predicted_eta_utc"] = "2020-01-01T00:00:00Z"
    elif corruption == "error":
        rows.loc[0, "eta_error_h"] += 1
    else:
        rows.loc[0, "was_clipped"] = not rows.loc[0, "was_clipped"]
    with pytest.raises(ValueError):
        validate_saved_predictions(rows, inputs, targets, plan)


def test_review_outputs_use_saved_validation_and_development_only(reviewed, evidence):
    output, summary = reviewed
    plan = read_json(evidence[1] / "split_manifest.json")
    expected_validation = set().union(*(set(fold["validation_ids"]) for fold in plan["folds"]))
    validation = pd.read_csv(output / "validation_errors.csv")
    assert len(validation) == 360
    for _, rows in validation.groupby("predictor"):
        assert set(rows.shipment_id) == expected_validation
        assert rows.shipment_id.is_unique
    responses = pd.read_csv(output / "stress_responses.csv")
    assert len(responses) == 2133
    assert set(responses.shipment_id) == set(plan["development_ids"])
    assert not set(responses.shipment_id).intersection(plan["holdout_ids"])
    assert "actual_delivery_utc" not in responses
    assert "port_disruption" not in responses
    assert summary["holdout_scored"] is summary["model_refitted"] is summary["predictor_changed"] is False
    assert summary["invalid_inputs_rejected"] == summary["invalid_input_cases"] == 8
    slices = pd.read_csv(output / "slice_metrics.csv")
    for predictor, rows in validation.groupby("predictor"):
        for dimension in ["route_id", "congestion_slice", "arrival_slice", "future_disruption_slice"]:
            selected = slices.loc[(slices.predictor == predictor) & (slices.dimension == dimension)]
            assert selected.n.sum() == 120
            assert (selected.small_sample == (selected.n < 10)).all()
            assert (selected.eta_mae_h.fillna(0) * selected.n).sum() / 120 == pytest.approx(rows.eta_error_h.abs().mean())
    selected = validation.loc[validation.predictor == summary["predictor"]]
    assert summary["selected_validation_eta_p95_abs_error_h"] == pytest.approx(selected.eta_error_h.abs().quantile(.95))
    largest = pd.read_csv(output / "largest_errors.csv")
    assert len(largest) == 10
    assert largest.absolute_eta_error_h.is_monotonic_decreasing
    assert largest.absolute_eta_error_h.iloc[0] == pytest.approx(selected.eta_error_h.abs().max())
    for name, checksum in read_json(output / "run_manifest.json")["files_sha256"].items():
        assert sha256(output / name) == checksum
    assert (output / "route_errors.png").read_bytes().startswith(b"\x89PNG")


def test_review_is_read_only_reproducible_and_never_predicts_holdout(reviewed, evidence, tmp_path, monkeypatch):
    from shipment_eta import stress, training
    before = {str(p): sha256(p) for directory in evidence for p in directory.iterdir()}
    plan = read_json(evidence[1] / "split_manifest.json")
    original_serve = stress.serve
    predicted_ids = []

    def observed_serve(model, metadata, inputs):
        predicted_ids.extend(inputs.shipment_id.tolist())
        assert set(inputs.shipment_id).issubset(plan["development_ids"])
        return original_serve(model, metadata, inputs)

    def no_fit(*args, **kwargs):
        pytest.fail("Review attempted model fitting")

    monkeypatch.setattr(stress, "serve", observed_serve)
    monkeypatch.setattr(training, "fit_candidate", no_fit)
    output = tmp_path / "repeated"
    summary = review_development(*evidence, output)
    assert summary == reviewed[1]
    assert set(predicted_ids) == set(plan["development_ids"])
    assert not set(predicted_ids).intersection(plan["holdout_ids"])
    for name in ["summary.json", "slice_metrics.csv", "stress_responses.csv", "largest_errors.csv"]:
        assert (output / name).read_bytes() == (reviewed[0] / name).read_bytes()
    assert before == {str(p): sha256(p) for directory in evidence for p in directory.iterdir()}
    with pytest.raises(FileExistsError):
        review_development(*evidence, output)


def test_changed_candidate_errors_rejected_before_model_loading(evidence, tmp_path, monkeypatch):
    from shipment_eta import robustness
    copied = tmp_path / "model"
    shutil.copytree(evidence[2], copied)
    path = copied / "candidate_validation_predictions.csv"
    path.write_bytes(path.read_bytes() + b"\n")

    def no_load(*args):
        pytest.fail("Model loaded before artifact integrity checks")

    monkeypatch.setattr(robustness, "load_trusted_model", no_load)
    with pytest.raises(ValueError, match="checksum"):
        review_development(evidence[0], evidence[1], copied, tmp_path / "output")


def test_review_cli_from_another_directory_and_repeat_failure(evidence, tmp_path):
    command = [sys.executable, "-W", "error", str(ROOT / "scripts/review_development.py"),
               "--data-dir", str(evidence[0]), "--stage5-dir", str(evidence[1]),
               "--model-dir", str(evidence[2]), "--output-dir", str(tmp_path / "result")]
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '"holdout_scored": false' in result.stdout
    repeated = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert repeated.returncode == 1
    assert "Output already exists" in repeated.stderr
