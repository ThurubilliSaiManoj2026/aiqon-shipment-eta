"""Deterministic feature construction from a strict checkpoint input schema."""

import numpy as np
import pandas as pd

from .schema import validate_inputs

BASE_FEATURES = ["planned_port_processing_h", "planned_inland_h", "port_congestion_index"]
PROPAGATION_FEATURES = BASE_FEATURES + [
    "port_arrival_deviation_h", "hours_until_first_dispatch", "dispatch_interval_h",
    "congestion_processing_h", "dispatch_phase_sin", "dispatch_phase_cos",
]
FEATURE_SETS = {"base": BASE_FEATURES, "propagation": PROPAGATION_FEATURES}
FEATURE_DEFINITIONS = {
    "planned_port_processing_h": "Scheduled cargo-readiness processing duration, hours.",
    "planned_inland_h": "Scheduled inland transit duration, hours.",
    "port_congestion_index": "Congestion observed at port arrival, dimensionless [0,1].",
    "port_arrival_deviation_h": "Actual minus scheduled port arrival, hours; signed.",
    "hours_until_first_dispatch": "max(0, planned first dispatch minus current port arrival), hours.",
    "dispatch_interval_h": "Known spacing between onward dispatch slots, hours.",
    "congestion_processing_h": "Current congestion multiplied by planned processing hours.",
    "dispatch_phase_sin": "Sine of current arrival phase relative to the known dispatch timetable.",
    "dispatch_phase_cos": "Cosine of current arrival phase relative to the known dispatch timetable.",
}


def build_features(inputs: pd.DataFrame, feature_set: str = "propagation") -> pd.DataFrame:
    """Return numeric features indexed by shipment ID; never fit statistics.

    Route identity is not encoded alongside route-constant numeric attributes.
    No target, actual future processing time, or simulator coefficient is used.
    Missing complete-schema inputs are rejected at this stage.
    """
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set: {feature_set}")
    validate_inputs(inputs)
    frame = inputs.set_index("shipment_id")
    result = frame[BASE_FEATURES].astype(float).copy()
    if feature_set == "propagation":
        arrival = frame.actual_port_arrival_utc
        first_dispatch = frame.scheduled_ready_utc
        result["port_arrival_deviation_h"] = (arrival - frame.scheduled_port_arrival_utc).dt.total_seconds() / 3600
        result["hours_until_first_dispatch"] = ((first_dispatch - arrival).dt.total_seconds() / 3600).clip(lower=0)
        result["dispatch_interval_h"] = frame.dispatch_interval_h
        result["congestion_processing_h"] = frame.port_congestion_index * frame.planned_port_processing_h
        # Integer modulo preserves the same timetable boundaries as the generator.
        intervals_ns = [pd.Timedelta(hours=h).value for h in frame.dispatch_interval_h]
        if any(interval <= 0 for interval in intervals_ns):
            raise ValueError("Dispatch interval is below timestamp resolution")
        phase = np.array([(a.value - s.value) % interval / interval
                          for a, s, interval in zip(arrival, first_dispatch, intervals_ns)])
        result["dispatch_phase_sin"] = np.sin(2 * np.pi * phase)
        result["dispatch_phase_cos"] = np.cos(2 * np.pi * phase)
    result = result[FEATURE_SETS[feature_set]].astype(float)
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError("Constructed features must be finite")
    return result
