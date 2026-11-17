"""Specification-based tests for synthetic journeys and failure handling."""

from copy import deepcopy
from pathlib import Path
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from shipment_eta.config import ConfigurationError, load_config, validate_config
from shipment_eta.dataset_io import export_dataset, read_table, sha256
from shipment_eta.schema import (AUDIT_FLAGS, FULL_COLUMNS, FUTURE_NUMBERS,
                                 FUTURE_TIMES, INPUT_COLUMNS, TARGETS,
                                 DataValidationError, checkpoint_inputs,
                                 validate_inputs, validate_journeys)
from shipment_eta.simulation import generate_journeys
from shipment_eta.timing import hours_between, next_dispatch

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def config():
    return load_config(ROOT / "config" / "simulation.toml")


@pytest.fixture(scope="module")
def journeys(config):
    return generate_journeys(config)


@pytest.mark.parametrize("offset_ns,expected_h", [
    (-3_600_000_000_000, 0), (0, 0), (1, 6),
    (6 * 3_600_000_000_000 - 1, 6),
    (6 * 3_600_000_000_000, 6), (6 * 3_600_000_000_000 + 1, 12),
])
def test_dispatch_nanosecond_boundaries(offset_ns, expected_h):
    first = pd.Timestamp("2025-01-02T12:00:00Z")
    ready = first + pd.Timedelta(offset_ns, unit="ns")
    assert next_dispatch(ready, first, 6) == first + pd.Timedelta(hours=expected_h)


@pytest.mark.parametrize("interval", [0, -1, float("nan"), float("inf"), True])
def test_invalid_dispatch_interval(interval):
    with pytest.raises(ValueError):
        next_dispatch("2025-01-01T00:00Z", "2025-01-01T00:00Z", interval)


def test_dispatch_rejects_naive_and_missing_timestamps():
    for invalid in [pd.NaT, "2025-01-01"]:
        with pytest.raises(ValueError, match="timezone-aware"):
            next_dispatch(invalid, "2025-01-01T00:00Z", 6)


@pytest.mark.parametrize("processing,expected_final,expected_signed", [(8, 26, -2), (20, 38, 10)])
def test_documented_recovery_and_amplification(processing, expected_final, expected_signed):
    anchor = pd.Timestamp("2025-01-01T00:00Z")
    readiness = anchor + pd.Timedelta(hours=6 + processing)
    dispatch = next_dispatch(readiness, anchor + pd.Timedelta(hours=12), 6)
    delivered = dispatch + pd.Timedelta(hours=8)
    assert hours_between(delivered, anchor) == expected_final
    assert hours_between(delivered, anchor + pd.Timedelta(hours=28)) == expected_signed


@pytest.mark.parametrize("path,value", [
    (("n_shipments",), 301), (("n_shipments",), True), (("seed",), -1),
    (("schema_version",), 2), (("scheduled_start_utc",), "2025-01-01"),
    (("scheduled_start_utc",), 2025), (("departure_spacing_h",), 0),
    (("port", "disruption_probability"), 1.5),
    (("voyage", "weather_beta_a"), 0), (("port", "noise_std_h"), -1),
    (("inland", "minimum_duration_h"), float("nan")),
    (("departure", "disruption_min_h"), 999),
])
def test_bad_config_is_rejected(config, path, value):
    altered = deepcopy(config)
    target = altered
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ConfigurationError):
        validate_config(altered)


def test_unknown_config_key_and_duplicate_routes(config):
    altered = deepcopy(config)
    altered["port"]["congeston_multiplier"] = 1.1
    with pytest.raises(ConfigurationError, match="unknown"):
        validate_config(altered)
    altered = deepcopy(config)
    altered["routes"][1]["route_id"] = altered["routes"][0]["route_id"]
    with pytest.raises(ConfigurationError, match="unique"):
        validate_config(altered)


def test_default_dataset_shape_and_independent_target_calculation(journeys, config):
    validate_journeys(journeys, config)
    assert len(journeys) == 300
    assert list(journeys.columns) == FULL_COLUMNS
    assert journeys.shipment_id.nunique() == 300
    assert journeys.route_id.value_counts().tolist() == [60] * 5
    for row in journeys.itertuples():
        expected_remaining = (row.actual_delivery_utc.value - row.actual_port_arrival_utc.value) / 3_600_000_000_000
        expected_signed = (row.actual_delivery_utc.value - row.scheduled_delivery_utc.value) / 3_600_000_000_000
        assert row.remaining_h == expected_remaining
        assert row.signed_arrival_deviation_h == expected_signed
        assert row.delay_magnitude_h == max(0, expected_signed)


def test_no_disruptions_or_noise_matches_hand_calculation(config):
    clean = deepcopy(config)
    clean["n_shipments"] = 5
    clean["departure"].update(normal_mean_h=0, normal_std_h=0, early_bound_h=0, disruption_probability=0)
    clean["voyage"].update(weather_effect_h=0, noise_std_fraction=0)
    clean["port"].update(base_multiplier=1, congestion_multiplier=0, noise_std_h=0, disruption_probability=0)
    clean["inland"].update(noise_std_fraction=0, disruption_probability=0)
    frame = generate_journeys(clean)
    assert (frame.actual_port_arrival_utc == frame.scheduled_port_arrival_utc).all()
    assert (frame.actual_dispatch_utc == frame.scheduled_ready_utc).all()
    np.testing.assert_allclose(frame.remaining_h, frame.planned_port_processing_h + frame.planned_inland_h)
    np.testing.assert_allclose(frame.signed_arrival_deviation_h, -frame.delivery_buffer_h)
    assert (frame.delay_magnitude_h == 0).all()
    assert frame[AUDIT_FLAGS].to_numpy().sum() == 0


def test_same_seed_reproduces_and_different_seed_changes(config, journeys):
    pd.testing.assert_frame_equal(generate_journeys(config), journeys, check_exact=True)
    altered = deepcopy(config)
    altered["seed"] += 1
    assert not generate_journeys(altered).actual_delivery_utc.equals(journeys.actual_delivery_utc)


def test_generation_does_not_mutate_config_or_global_rng(config):
    before_config = deepcopy(config)
    state = np.random.get_state()
    generate_journeys(config)
    after = np.random.get_state()
    assert config == before_config
    assert state[0] == after[0]
    np.testing.assert_array_equal(state[1], after[1])
    assert state[2:] == after[2:]


@pytest.mark.parametrize("seed", [0, 7, 2026])
def test_invariants_across_additional_seeds(config, seed):
    altered = deepcopy(config)
    altered["seed"] = seed
    validate_journeys(generate_journeys(altered), altered)


def test_future_changes_cannot_change_checkpoint_inputs(journeys):
    original = checkpoint_inputs(journeys)
    changed = journeys.copy(deep=True)
    for name in FUTURE_TIMES:
        changed[name] = changed[name] + pd.Timedelta(days=90)
    for name in FUTURE_NUMBERS + TARGETS + AUDIT_FLAGS:
        changed[name] = -999
    pd.testing.assert_frame_equal(checkpoint_inputs(changed), original)
    assert list(original.columns) == INPUT_COLUMNS
    original.loc[0, "port_congestion_index"] = 0
    assert journeys.loc[0, "port_congestion_index"] != 0  # verifies the view is an independent copy


def test_input_validator_rejects_target_column_and_missing_checkpoint(journeys):
    inputs = checkpoint_inputs(journeys)
    with pytest.raises(DataValidationError, match="Column mismatch"):
        validate_inputs(inputs.assign(remaining_h=1))
    inputs.loc[0, "actual_port_arrival_utc"] = pd.NaT
    with pytest.raises(DataValidationError, match="missing"):
        validate_inputs(inputs)


@pytest.mark.parametrize("column,value,message", [
    ("shipment_id", "SHP-000002", "Duplicate shipment"),
    ("remaining_h", -1.0, "remaining_h"),
    ("port_congestion_index", 1.1, "in \\[0,1\\]"),
    ("actual_inland_h", float("inf"), "finite"),
    ("destination", "wrong", "route destination"),
])
def test_corrupted_records_rejected(journeys, config, column, value, message):
    broken = journeys.copy(deep=True)
    broken.loc[0, column] = value
    with pytest.raises(DataValidationError, match=message):
        validate_journeys(broken, config)


def test_invalid_order_and_future_checkpoint_rejected(journeys, config):
    broken = journeys.copy(deep=True)
    broken.loc[0, "actual_ready_utc"] = broken.loc[0, "actual_port_arrival_utc"] - pd.Timedelta(hours=1)
    with pytest.raises(DataValidationError, match="Invalid order"):
        validate_journeys(broken, config)
    inputs = checkpoint_inputs(journeys)
    inputs.loc[0, "as_of_utc"] += pd.Timedelta(hours=1)
    with pytest.raises(DataValidationError, match="Checkpoint"):
        validate_inputs(inputs)


def test_export_round_trip_hashes_separation_and_overwrite_protection(tmp_path, config):
    destination = tmp_path / "generated"
    audit = export_dataset(ROOT / "config/simulation.toml", destination)
    assert audit["n_shipments"] == 300
    assert audit["missing_values"] == 0
    assert sum(audit["arrival_counts"].values()) == 300
    assert audit["split_feasibility"]["holdout_shipments"] == 60
    restored = read_table(destination / "journeys.csv")
    validate_journeys(restored, config)
    inputs = read_table(destination / "checkpoint_inputs.csv")
    validate_inputs(inputs)
    assert not set(TARGETS + FUTURE_TIMES).intersection(inputs.columns)
    targets = read_table(destination / "targets.csv")
    assert targets.shipment_id.equals(inputs.shipment_id)
    manifest = json.loads((destination / "manifest.json").read_text())
    for name, checksum in manifest["files_sha256"].items():
        assert sha256(destination / name) == checksum
    original_bytes = (destination / "journeys.csv").read_bytes()
    with pytest.raises(FileExistsError, match="already exists"):
        export_dataset(ROOT / "config/simulation.toml", destination)
    assert (destination / "journeys.csv").read_bytes() == original_bytes
    repeated = tmp_path / "repeated"
    export_dataset(ROOT / "config/simulation.toml", repeated)
    for name in ["journeys.csv", "checkpoint_inputs.csv", "targets.csv", "audit.json"]:
        assert (destination / name).read_bytes() == (repeated / name).read_bytes()


def test_csv_naive_timestamps_rejected(tmp_path):
    bad = tmp_path / "naive.csv"
    bad.write_text("actual_port_arrival_utc\n2025-01-01T00:00:00\n", encoding="utf-8")
    with pytest.raises(ValueError, match="explicit UTC"):
        read_table(bad)


def test_cli_end_to_end_and_error_exit(tmp_path):
    command = [sys.executable, str(ROOT / "scripts/generate_data.py"), "--output-dir", str(tmp_path / "run")]
    completed = subprocess.run(command, capture_output=True, text=True, cwd=tmp_path)
    assert completed.returncode == 0, completed.stderr
    assert "Validated dataset written" in completed.stdout
    assert (tmp_path / "run/manifest.json").is_file()
    repeated = subprocess.run(command, capture_output=True, text=True, cwd=tmp_path)
    assert repeated.returncode == 1
    assert "Output already exists" in repeated.stderr
    failed = subprocess.run(command + ["--config", str(tmp_path / "missing.toml"),
                                      "--output-dir", str(tmp_path / "unused")],
                            capture_output=True, text=True, cwd=tmp_path)
    assert failed.returncode == 1
    assert not (tmp_path / "unused").exists()
