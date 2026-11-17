"""Exact model arithmetic relative to a declared reference, without causal claims."""

import numpy as np

from .evaluation import BASELINES, baseline_prediction
from .features import FEATURE_DEFINITIONS, FEATURE_SETS, build_features
from .modeling import predict_candidate
from .schema import validate_inputs

GROUPS = {
    "port_processing": ("Port processing and congestion", ["planned_port_processing_h",
                         "port_congestion_index", "congestion_processing_h"]),
    "inland_plan": ("Inland travel plan", ["planned_inland_h"]),
    "upstream_arrival": ("Upstream arrival deviation", ["port_arrival_deviation_h"]),
    "dispatch_schedule": ("Dispatch timetable", ["hours_until_first_dispatch",
                          "dispatch_interval_h", "dispatch_phase_sin", "dispatch_phase_cos"]),
}
FEATURE_GROUP = {feature: group for group, (_, features) in GROUPS.items() for feature in features}
BASELINE_LABELS = {"scheduled_remaining": "Time from checkpoint to scheduled delivery",
                   "upstream_arrival": "Signed port-arrival deviation"}
INTERPRETATION = (
    "Contributions explain this predictor relative to its stated reference; they are not "
    "causal effects, observed stage durations, delay probabilities, or calibrated uncertainty. "
    "Feature contributions add to raw remaining time, not to clipped delay magnitude."
)


def predict_and_explain(model, metadata, inputs):
    """Return served predictions and a reconstructable explanation for every row."""
    validate_inputs(inputs)
    predictor = metadata["validation_recommended_predictor"]
    is_ridge = predictor == metadata["selection"]["candidate"]
    if is_ridge:
        group = metadata["selection"]["feature_set"]
        features = build_features(inputs, group)
        names = FEATURE_SETS[group]
        scaler, ridge = model.named_steps["scale"], model.named_steps["regressor"]
        if list(scaler.feature_names_in_) != names or metadata["feature_columns"] != names:
            raise ValueError("Explanation feature schema differs from the fitted model")
        means, scales, coefficients = scaler.mean_, scaler.scale_, ridge.coef_
        reference_h = float(ridge.intercept_)
        if not all(np.isfinite(v).all() for v in (means, scales, coefficients)) or not np.isfinite(reference_h):
            raise ValueError("Explanation parameters must be finite")
        if (scales <= 0).any():
            raise ValueError("Explanation scales must be positive")
        values = features.to_numpy(dtype=float)
        standardized = (values - means) / scales
        contributions = standardized * coefficients
        predictions = predict_candidate(model, inputs)
        reference = "Fitted training-mean feature vector; not necessarily a physically realizable shipment."
        method = "exact_linear_training_mean_reference"
        ranges = metadata["training_feature_ranges"]
        for name in names:
            low, high = ranges[name]["min"], ranges[name]["max"]
            if not np.isfinite([low, high]).all() or low > high:
                raise ValueError("Invalid training feature range")
    elif predictor in BASELINES:
        predictions = baseline_prediction(inputs, predictor)
        reference_h, reference = 0.0, "Zero remaining hours; direct algebraic decomposition of the rule."
        method = "exact_baseline_identity"
    else:
        raise ValueError("Unrecognized saved validation recommendation")

    records = []
    for position, row in enumerate(inputs.itertuples(index=False)):
        prediction = predictions.loc[row.shipment_id]
        terms, range_flags = [], []
        if is_ridge:
            for index, name in enumerate(names):
                value = float(values[position, index])
                terms.append({"term": name, "group": FEATURE_GROUP[name],
                              "description": FEATURE_DEFINITIONS[name], "value": value,
                              "reference_value": float(means[index]), "scale": float(scales[index]),
                              "coefficient_standardized_h": float(coefficients[index]),
                              "contribution_h": float(contributions[position, index])})
                low, high = ranges[name]["min"], ranges[name]["max"]
                if value < low - 1e-9 or value > high + 1e-9:
                    range_flags.append({"feature": name, "value": value,
                                        "training_min": low, "training_max": high})
        else:
            terms.append({"term": "scheduled_remaining_h", "group": "scheduled_remaining",
                          "description": "Scheduled delivery minus current checkpoint, hours.",
                          "value": (row.scheduled_delivery_utc - row.actual_port_arrival_utc).value / 3.6e12,
                          "reference_value": 0.0, "scale": 1.0, "coefficient_standardized_h": 1.0})
            if predictor == "carry_forward":
                terms.append({"term": "port_arrival_deviation_h", "group": "upstream_arrival",
                              "description": "Actual minus scheduled port arrival, hours.",
                              "value": (row.actual_port_arrival_utc - row.scheduled_port_arrival_utc).value / 3.6e12,
                              "reference_value": 0.0, "scale": 1.0, "coefficient_standardized_h": 1.0})
            for term in terms:
                term["contribution_h"] = term["value"]

        raw_reconstructed = reference_h + sum(term["contribution_h"] for term in terms)
        raw_error = abs(raw_reconstructed - float(prediction.raw_remaining_h))
        if not np.isclose(raw_reconstructed, prediction.raw_remaining_h, rtol=1e-12, atol=1e-8):
            raise ValueError("Explanation does not reconstruct the raw prediction")
        deadline_offset = (row.actual_port_arrival_utc - row.scheduled_delivery_utc).value / 3.6e12
        signed_reconstructed = deadline_offset + float(prediction.predicted_remaining_h)
        if not np.isclose(signed_reconstructed, prediction.predicted_signed_deviation_h, rtol=0, atol=1e-8):
            raise ValueError("Explanation does not reconstruct the ETA/deadline relationship")

        grouped = {}
        for term in terms:
            grouped[term["group"]] = grouped.get(term["group"], 0.0) + term["contribution_h"]
        groups = [{"group": key, "label": GROUPS[key][0] if is_ridge else BASELINE_LABELS[key],
                   "contribution_h": float(value)} for key, value in grouped.items()]
        groups.sort(key=lambda item: (-abs(item["contribution_h"]), item["group"]))
        records.append({
            "shipment_id": row.shipment_id, "predictor": predictor,
            "as_of_utc": row.as_of_utc.isoformat(),
            "scheduled_delivery_utc": row.scheduled_delivery_utc.isoformat(),
            "predicted_eta_utc": prediction.predicted_eta_utc.isoformat(),
            "reference_remaining_h": reference_h,
            "raw_remaining_h": float(prediction.raw_remaining_h),
            "remaining_adjustment_h": float(prediction.remaining_adjustment_h),
            "predicted_remaining_h": float(prediction.predicted_remaining_h),
            "remaining_was_clipped": bool(prediction.was_clipped),
            "deadline_offset_h": float(deadline_offset),
            "predicted_signed_deviation_h": float(prediction.predicted_signed_deviation_h),
            "predicted_delay_magnitude_h": float(prediction.predicted_delay_magnitude_h),
            "delay_floor_adjustment_h": max(0.0, -float(prediction.predicted_signed_deviation_h)),
            "reconstruction_abs_error_h": raw_error,
            "feature_terms": terms, "group_contributions": groups,
            "training_range_flags": range_flags,
        })
    document = {"schema_version": 1, "predictor": predictor, "method": method,
                "reference": reference, "interpretation": INTERPRETATION,
                "range_check_applicable": is_ridge,
                "range_check_limit": "Per-feature min/max only; no flag does not establish reliability.",
                "records": records}
    return predictions, document
