"""An explicit prediction-input allowlist and independent journey checks."""

import numpy as np
import pandas as pd

from .config import validate_config
from .timing import next_dispatch

IDENTIFIERS = ["shipment_id", "route_id", "origin", "destination_port", "destination"]
INPUT_TIMES = ["scheduled_departure_utc", "scheduled_port_arrival_utc",
               "scheduled_ready_utc", "scheduled_delivery_utc", "actual_departure_utc",
               "actual_port_arrival_utc", "as_of_utc"]
INPUT_NUMBERS = ["planned_voyage_h", "planned_port_processing_h", "planned_inland_h",
                 "delivery_buffer_h", "dispatch_interval_h", "voyage_weather_index",
                 "port_congestion_index", "actual_voyage_h"]
INPUT_COLUMNS = IDENTIFIERS + INPUT_TIMES + INPUT_NUMBERS
FUTURE_TIMES = ["actual_ready_utc", "actual_dispatch_utc", "actual_delivery_utc"]
FUTURE_NUMBERS = ["actual_port_processing_h", "dispatch_wait_h", "actual_inland_h"]
TARGETS = ["remaining_h", "signed_arrival_deviation_h", "delay_magnitude_h"]
AUDIT_FLAGS = ["departure_disruption", "port_disruption", "inland_disruption"]
FULL_COLUMNS = INPUT_COLUMNS + FUTURE_TIMES + FUTURE_NUMBERS + TARGETS + AUDIT_FLAGS
TARGET_COLUMNS = ["shipment_id", "actual_delivery_utc", *TARGETS]


class DataValidationError(ValueError):
    """A shipment table violates the documented data contract."""


def _require(condition, message):
    if not bool(condition):
        raise DataValidationError(message)


def _basic(frame, columns, times, numbers):
    _require(isinstance(frame, pd.DataFrame), "Expected a pandas DataFrame")
    _require(not frame.empty, "Shipment table must not be empty")
    _require(frame.columns.is_unique, "Duplicate column names")
    _require(set(frame.columns) == set(columns),
             f"Column mismatch: missing={sorted(set(columns)-set(frame.columns))}, "
             f"unexpected={sorted(set(frame.columns)-set(columns))}")
    _require(not frame.isna().any().any(), "Complete base dataset contains missing values")
    for name in IDENTIFIERS:
        _require(frame[name].map(lambda v: isinstance(v, str) and bool(v.strip())).all(),
                 f"{name} must contain nonempty strings")
    _require(frame.shipment_id.is_unique, "Duplicate shipment IDs")
    for name in times:
        dtype = frame[name].dtype
        _require(isinstance(dtype, pd.DatetimeTZDtype) and str(dtype.tz) == "UTC",
                 f"{name} must have timezone-aware UTC dtype")
    for name in numbers:
        _require(pd.api.types.is_numeric_dtype(frame[name]) and
                 not pd.api.types.is_bool_dtype(frame[name]), f"{name} must be numeric")
        _require(np.isfinite(frame[name].to_numpy(dtype=float)).all(), f"{name} must be finite")


def _close(actual, expected, name):
    _require(np.allclose(actual, expected, rtol=0, atol=1e-9), f"Inconsistent {name}")


def _duration(frame, end, start):
    return (frame[end] - frame[start]).dt.total_seconds().to_numpy() / 3600


def validate_inputs(frame: pd.DataFrame) -> None:
    """Validate complete checkpoint inputs; optional missing-input support is later work."""
    _basic(frame, INPUT_COLUMNS, INPUT_TIMES, INPUT_NUMBERS)
    for name in ["planned_voyage_h", "planned_port_processing_h", "planned_inland_h",
                 "dispatch_interval_h", "actual_voyage_h"]:
        _require((frame[name] > 0).all(), f"{name} must be positive")
    _require((frame.delivery_buffer_h >= 0).all(), "Negative delivery buffer")
    for name in ["voyage_weather_index", "port_congestion_index"]:
        _require(frame[name].between(0, 1).all(), f"{name} must be in [0,1]")
    for left, right in [("scheduled_departure_utc", "scheduled_port_arrival_utc"),
                        ("scheduled_port_arrival_utc", "scheduled_ready_utc"),
                        ("scheduled_ready_utc", "scheduled_delivery_utc"),
                        ("actual_departure_utc", "actual_port_arrival_utc")]:
        _require((frame[left] < frame[right]).all(), f"Invalid order: {left}, {right}")
    _require((frame.as_of_utc == frame.actual_port_arrival_utc).all(), "Checkpoint must equal port arrival")
    _close(_duration(frame, "scheduled_port_arrival_utc", "scheduled_departure_utc"),
           frame.planned_voyage_h, "planned voyage")
    _close(_duration(frame, "scheduled_ready_utc", "scheduled_port_arrival_utc"),
           frame.planned_port_processing_h, "planned processing")
    _close(_duration(frame, "scheduled_delivery_utc", "scheduled_ready_utc"),
           frame.planned_inland_h + frame.delivery_buffer_h, "planned delivery")
    _close(_duration(frame, "actual_port_arrival_utc", "actual_departure_utc"),
           frame.actual_voyage_h, "actual voyage")


def checkpoint_inputs(journeys: pd.DataFrame) -> pd.DataFrame:
    """Copy only known-at-checkpoint fields, regardless of future-column values."""
    missing = set(INPUT_COLUMNS) - set(journeys.columns)
    _require(not missing, f"Missing input columns: {sorted(missing)}")
    result = journeys.loc[:, INPUT_COLUMNS].copy(deep=True)
    validate_inputs(result)
    return result


def validate_journeys(frame: pd.DataFrame, config: dict) -> None:
    config = validate_config(config)
    _basic(frame, FULL_COLUMNS, INPUT_TIMES + FUTURE_TIMES,
           INPUT_NUMBERS + FUTURE_NUMBERS + TARGETS + AUDIT_FLAGS)
    validate_inputs(frame.loc[:, INPUT_COLUMNS])
    _require(len(frame) == config["n_shipments"], "Unexpected shipment count")
    expected_departures = pd.DatetimeIndex([
        pd.Timestamp(config["scheduled_start_utc"]).tz_convert("UTC")
        + pd.Timedelta(hours=i * config["departure_spacing_h"])
        for i in range(config["n_shipments"])
    ])
    observed_departures = pd.DatetimeIndex(frame.scheduled_departure_utc.sort_values())
    _require((observed_departures == expected_departures).all(), "Departure schedule differs from config")
    counts = frame.route_id.value_counts().to_dict()
    expected = {r["route_id"]: config["n_shipments"] // len(config["routes"]) for r in config["routes"]}
    _require(counts == expected, "Route counts do not match balanced configuration")
    for route in config["routes"]:
        rows = frame.loc[frame.route_id == route["route_id"]]
        for name in ["origin", "destination_port", "destination"]:
            _require((rows[name] == route[name]).all(), f"Inconsistent route {name}")
        for column, key in {"planned_voyage_h": "voyage_h",
                            "planned_port_processing_h": "port_processing_h",
                            "planned_inland_h": "inland_h", "delivery_buffer_h": "delivery_buffer_h",
                            "dispatch_interval_h": "dispatch_interval_h"}.items():
            _close(rows[column], route[key], column)
    for left, right, inclusive in [("actual_port_arrival_utc", "actual_ready_utc", False),
                                   ("actual_ready_utc", "actual_dispatch_utc", True),
                                   ("actual_dispatch_utc", "actual_delivery_utc", False)]:
        order = frame[left] <= frame[right] if inclusive else frame[left] < frame[right]
        _require(order.all(), f"Invalid order: {left}, {right}")
    for row in frame.itertuples(index=False):
        _require(row.actual_dispatch_utc == next_dispatch(row.actual_ready_utc,
                  row.scheduled_ready_utc, row.dispatch_interval_h), "Dispatch is not the first eligible slot")
    for col, end, start in [
        ("actual_port_processing_h", "actual_ready_utc", "actual_port_arrival_utc"),
        ("dispatch_wait_h", "actual_dispatch_utc", "actual_ready_utc"),
        ("actual_inland_h", "actual_delivery_utc", "actual_dispatch_utc"),
        ("remaining_h", "actual_delivery_utc", "actual_port_arrival_utc"),
        ("signed_arrival_deviation_h", "actual_delivery_utc", "scheduled_delivery_utc"),
    ]:
        _close(frame[col], _duration(frame, end, start), col)
    _close(frame.delay_magnitude_h, np.maximum(0, frame.signed_arrival_deviation_h), "delay magnitude")
    for name in ["actual_port_processing_h", "actual_inland_h", "remaining_h"]:
        _require((frame[name] > 0).all(), f"{name} must be positive")
    for name in ["dispatch_wait_h", "delay_magnitude_h"]:
        _require((frame[name] >= 0).all(), f"{name} must be nonnegative")
    for name in AUDIT_FLAGS:
        _require(frame[name].isin([0, 1]).all(), f"{name} must contain zero/one flags")
    for actual, minimum, name in [
        (frame.actual_voyage_h, frame.planned_voyage_h * config["voyage"]["minimum_duration_fraction"], "voyage floor"),
        (frame.actual_port_processing_h, config["port"]["minimum_duration_h"], "processing floor"),
        (frame.actual_inland_h, config["inland"]["minimum_duration_h"], "inland floor"),
        (_duration(frame, "actual_departure_utc", "scheduled_departure_utc"), config["departure"]["early_bound_h"], "departure bound"),
    ]:
        _require((actual >= minimum - 1e-9).all(), f"Violation of {name}")
