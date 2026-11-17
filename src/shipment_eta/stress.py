"""Coherent input perturbations of a fixed predictor, without outcome labels."""

import numpy as np
import pandas as pd

from .evaluation import BASELINES, baseline_prediction
from .features import build_features
from .modeling import predict_candidate
from .schema import validate_inputs

CONGESTION_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
ARRIVAL_SHIFTS_H = (1.0, 6.0, 12.0)
INLAND_SHIFT_H = 6.0
DIRECTION_TOLERANCE_H = 1e-8


def serve(model, metadata, inputs):
    predictor = metadata["validation_recommended_predictor"]
    if predictor == metadata["selection"]["candidate"]:
        return predict_candidate(model, inputs)
    if predictor in BASELINES:
        return baseline_prediction(inputs, predictor)
    raise ValueError("Unknown saved predictor")


def perturb(inputs, family, amount):
    """Change raw inputs together so time/duration identities remain true."""
    validate_inputs(inputs)
    changed = inputs.copy(deep=True)
    if family == "congestion":
        if not np.isfinite(amount) or not 0 <= amount <= 1:
            raise ValueError("Congestion must be in [0,1]")
        changed["port_congestion_index"] = float(amount)
    elif family == "inland_plan":
        if not np.isfinite(amount) or amount <= 0:
            raise ValueError("Inland shift must be positive")
        changed["planned_inland_h"] += amount
        changed["scheduled_delivery_utc"] += pd.Timedelta(hours=amount)
    elif family == "later_arrival":
        if not np.isfinite(amount) or amount <= 0:
            raise ValueError("Arrival shift must be positive")
        changed["actual_port_arrival_utc"] += pd.Timedelta(hours=amount)
        changed["as_of_utc"] += pd.Timedelta(hours=amount)
        changed["actual_voyage_h"] += amount
    else:
        raise ValueError("Unknown perturbation family")
    validate_inputs(changed)
    return changed


def outside_training_ranges(inputs, metadata):
    if metadata["validation_recommended_predictor"] in BASELINES:
        return np.zeros(len(inputs), dtype=bool)
    features = build_features(inputs, metadata["selection"]["feature_set"])
    flagged = np.zeros(len(inputs), dtype=bool)
    for name in features:
        bounds = metadata["training_feature_ranges"][name]
        flagged |= ((features[name].to_numpy() < bounds["min"] - 1e-9) |
                    (features[name].to_numpy() > bounds["max"] + 1e-9))
    return flagged


def stress_responses(model, metadata, inputs):
    """Return per-row responses and directional checks on fixed development IDs."""
    validate_inputs(inputs)
    if inputs.shipment_id.tolist() != metadata["training_ids"]:
        raise ValueError("Stress inputs must be exactly the saved development training IDs in order")
    original = serve(model, metadata, inputs)
    records, checks = [], []
    previous = None
    scenarios = [("congestion", value) for value in CONGESTION_GRID]
    scenarios += [("inland_plan", INLAND_SHIFT_H)]
    scenarios += [("later_arrival", value) for value in ARRIVAL_SHIFTS_H]
    for family, amount in scenarios:
        changed = perturb(inputs, family, amount)
        prediction = serve(model, metadata, changed)
        delta_remaining = prediction.predicted_remaining_h - original.predicted_remaining_h
        delta_eta = (prediction.predicted_eta_utc - original.predicted_eta_utc).dt.total_seconds() / 3600
        flagged = outside_training_ranges(changed, metadata)
        if family == "congestion":
            directional_delta = None if previous is None else prediction.predicted_remaining_h - previous.predicted_remaining_h
            previous = prediction
            compared_quantity = "remaining_hours_vs_previous_congestion_level"
        elif family == "inland_plan":
            directional_delta, compared_quantity = delta_remaining, "remaining_hours_vs_original"
        else:
            directional_delta, compared_quantity = delta_eta, "final_eta_hours_vs_original"
        n_checks = 0 if directional_delta is None else len(inputs)
        decreases = 0 if directional_delta is None else int((directional_delta < -DIRECTION_TOLERANCE_H).sum())
        checks.append({"family": family, "setting": amount, "n": len(inputs),
                       "compared_quantity": compared_quantity, "directional_comparisons": n_checks,
                       "directional_decreases": decreases,
                       "minimum_directional_change_h": None if directional_delta is None else float(directional_delta.min()),
                       "mean_remaining_change_h": float(delta_remaining.mean()),
                       "mean_eta_change_h": float(delta_eta.mean()),
                       "outside_training_range_rows": int(flagged.sum()),
                       "remaining_clipped_count": int(prediction.was_clipped.sum())})
        table = pd.DataFrame({"shipment_id": inputs.shipment_id.to_numpy(),
                              "family": family, "setting": amount,
                              "original_remaining_h": original.predicted_remaining_h.to_numpy(),
                              "predicted_remaining_h": prediction.predicted_remaining_h.to_numpy(),
                              "remaining_change_h": delta_remaining.to_numpy(),
                              "eta_change_h": delta_eta.to_numpy(),
                              "delay_change_h": (prediction.predicted_delay_magnitude_h - original.predicted_delay_magnitude_h).to_numpy(),
                              "outside_training_range": flagged,
                              "was_clipped": prediction.was_clipped.to_numpy()})
        records.append(table)
    return pd.concat(records, ignore_index=True), checks


def invalid_input_checks(model, metadata, inputs):
    """These are hard contract checks, not model quality thresholds."""
    source = inputs.head(1).copy().reset_index(drop=True)
    variants = {}
    bad = source.copy(); bad["port_congestion_index"] = np.nan
    variants["missing_congestion"] = bad
    bad = source.copy(); bad["port_congestion_index"] = np.inf
    variants["infinite_congestion"] = bad
    bad = source.copy(); bad["port_congestion_index"] = 1.1
    variants["out_of_bounds_congestion"] = bad
    variants["future_outcome_column"] = source.assign(actual_delivery_utc=source.as_of_utc + pd.Timedelta(days=1))
    bad = source.copy(); bad["planned_inland_h"] = -1.0
    variants["negative_inland_duration"] = bad
    variants["duplicate_shipment_id"] = pd.concat([source, source], ignore_index=True)
    bad = source.copy(); bad["as_of_utc"] = bad.as_of_utc.dt.tz_localize(None)
    variants["naive_checkpoint_timestamp"] = bad
    bad = source.copy(); bad["actual_voyage_h"] += 1.0
    variants["inconsistent_voyage_duration"] = bad
    records = []
    for name, frame in variants.items():
        try:
            serve(model, metadata, frame)
        except ValueError as exc:
            records.append({"case": name, "rejected": True, "reason": str(exc)})
        else:
            raise ValueError(f"Invalid input unexpectedly accepted: {name}")
    return pd.DataFrame(records)
