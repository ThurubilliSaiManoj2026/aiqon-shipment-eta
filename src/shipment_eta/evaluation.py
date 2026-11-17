"""Aligned targets, physically constrained predictions, and hour-based metrics."""

import numpy as np
import pandas as pd

from .schema import TARGET_COLUMNS, TARGETS, validate_inputs

BASELINES = ("scheduled", "carry_forward")


def align_targets(inputs: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """Validate targets and join by ID, never by accidental CSV row position."""
    validate_inputs(inputs)
    if not targets.columns.is_unique or set(targets.columns) != set(TARGET_COLUMNS):
        raise ValueError("Target table has incorrect columns")
    if targets.isna().any().any() or not targets.shipment_id.is_unique:
        raise ValueError("Targets must have unique IDs and no missing values")
    if set(targets.shipment_id) != set(inputs.shipment_id):
        raise ValueError("Input and target shipment IDs differ")
    dtype = targets.actual_delivery_utc.dtype
    if not isinstance(dtype, pd.DatetimeTZDtype) or str(dtype.tz) != "UTC":
        raise ValueError("Target delivery timestamp must be timezone-aware UTC")
    for name in TARGETS:
        if not pd.api.types.is_numeric_dtype(targets[name]) or pd.api.types.is_bool_dtype(targets[name]):
            raise ValueError(f"Target {name} must be numeric")
    if not np.isfinite(targets[TARGETS].to_numpy(dtype=float)).all():
        raise ValueError("Targets must be finite")
    source = inputs.set_index("shipment_id")
    aligned = targets.set_index("shipment_id").loc[source.index].copy()
    remaining = (aligned.actual_delivery_utc - source.actual_port_arrival_utc).dt.total_seconds() / 3600
    signed = (aligned.actual_delivery_utc - source.scheduled_delivery_utc).dt.total_seconds() / 3600
    if not (remaining > 0).all():
        raise ValueError("Target delivery must be after the checkpoint")
    for name, expected in [("remaining_h", remaining), ("signed_arrival_deviation_h", signed),
                            ("delay_magnitude_h", signed.clip(lower=0))]:
        if not np.allclose(aligned[name], expected, rtol=0, atol=1e-9):
            raise ValueError(f"Target {name} is inconsistent with timestamps")
    return aligned


def prediction_from_remaining(inputs: pd.DataFrame, raw_remaining: pd.Series) -> pd.DataFrame:
    """Reconstruct ETA and delay; retain an explicit nonnegative-time adjustment.

    The supplied Series must be indexed in input shipment-ID order. This catches
    a common silent error when applying predictions to shuffled records.
    """
    validate_inputs(inputs)
    source = inputs.set_index("shipment_id")
    if not isinstance(raw_remaining, pd.Series) or not raw_remaining.index.equals(source.index):
        raise ValueError("Predictions must be a Series in input shipment-ID order")
    if not pd.api.types.is_numeric_dtype(raw_remaining) or pd.api.types.is_bool_dtype(raw_remaining):
        raise ValueError("Predictions must be numeric")
    if not np.isfinite(raw_remaining.to_numpy(dtype=float)).all():
        raise ValueError("Predictions must be finite")
    raw = raw_remaining.astype(float)
    served = raw.clip(lower=0)
    eta = source.actual_port_arrival_utc + pd.to_timedelta(served, unit="h")
    signed = (eta - source.scheduled_delivery_utc).dt.total_seconds() / 3600
    result = pd.DataFrame(index=source.index)
    result["raw_remaining_h"] = raw
    result["predicted_remaining_h"] = served
    result["remaining_adjustment_h"] = served - raw
    result["was_clipped"] = raw < 0
    result["predicted_eta_utc"] = eta
    result["predicted_signed_deviation_h"] = signed
    result["predicted_delay_magnitude_h"] = signed.clip(lower=0)
    return result


def baseline_prediction(inputs: pd.DataFrame, name: str) -> pd.DataFrame:
    if name not in BASELINES:
        raise ValueError(f"Unknown baseline: {name}")
    validate_inputs(inputs)
    source = inputs.set_index("shipment_id")
    raw = (source.scheduled_delivery_utc - source.actual_port_arrival_utc).dt.total_seconds() / 3600
    if name == "carry_forward":
        raw += (source.actual_port_arrival_utc - source.scheduled_port_arrival_utc).dt.total_seconds() / 3600
    return prediction_from_remaining(inputs, raw)


def prediction_errors(inputs: pd.DataFrame, targets: pd.DataFrame, prediction: pd.DataFrame) -> pd.DataFrame:
    aligned = align_targets(inputs, targets)
    if not prediction.index.equals(aligned.index):
        raise ValueError("Prediction and target ID order differ")
    required = {"predicted_eta_utc", "predicted_delay_magnitude_h", "was_clipped"}
    if not required.issubset(prediction.columns):
        raise ValueError("Missing prediction output columns")
    if prediction[list(required)].isna().any().any():
        raise ValueError("Prediction outputs contain missing values")
    if not pd.api.types.is_bool_dtype(prediction.was_clipped):
        raise ValueError("was_clipped must be boolean")
    eta_dtype = prediction.predicted_eta_utc.dtype
    if not isinstance(eta_dtype, pd.DatetimeTZDtype) or str(eta_dtype.tz) != "UTC":
        raise ValueError("Predicted ETA must be timezone-aware UTC")
    source = inputs.set_index("shipment_id")
    if not (prediction.predicted_eta_utc >= source.actual_port_arrival_utc).all():
        raise ValueError("Predicted ETA precedes the checkpoint")
    expected_delay = ((prediction.predicted_eta_utc - source.scheduled_delivery_utc).dt.total_seconds() / 3600).clip(lower=0)
    if not np.allclose(prediction.predicted_delay_magnitude_h, expected_delay, rtol=0, atol=1e-9):
        raise ValueError("Predicted delay is inconsistent with ETA")
    result = pd.DataFrame(index=aligned.index)
    result["eta_error_h"] = (prediction.predicted_eta_utc - aligned.actual_delivery_utc).dt.total_seconds() / 3600
    result["delay_error_h"] = prediction.predicted_delay_magnitude_h - aligned.delay_magnitude_h
    result["was_clipped"] = prediction.was_clipped
    return result


def summarize_errors(errors: pd.DataFrame) -> dict:
    if errors.empty:
        raise ValueError("Cannot score an empty prediction set")
    numeric = errors[["eta_error_h", "delay_error_h"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("Metric inputs must be finite")
    if not pd.api.types.is_bool_dtype(errors.was_clipped):
        raise ValueError("was_clipped must be boolean")
    return {"n": len(errors), "eta_mae_h": float(np.mean(np.abs(numeric[:, 0]))),
            "eta_rmse_h": float(np.sqrt(np.mean(numeric[:, 0] ** 2))),
            "eta_bias_h": float(np.mean(numeric[:, 0])),
            "delay_mae_h": float(np.mean(np.abs(numeric[:, 1]))),
            "clipped_count": int(errors.was_clipped.sum())}
