"""Explicit chronological partitions with delivery-label availability checks."""

from pathlib import Path
import tomllib

from .evaluation import align_targets

EVALUATION_KEYS = {"schema_version", "holdout_size", "n_validation_folds",
                   "validation_size", "minimum_training_size"}


def validate_evaluation_config(config: dict) -> dict:
    if not isinstance(config, dict) or set(config) != EVALUATION_KEYS:
        raise ValueError(f"Evaluation config requires exactly {sorted(EVALUATION_KEYS)}")
    if any(type(v) is not int or v < 1 for v in config.values()):
        raise ValueError("Evaluation parameters must be positive integers")
    if config["schema_version"] != 1:
        raise ValueError("Only evaluation schema_version=1 is supported")
    return dict(config)


def load_evaluation_config(path: str | Path) -> dict:
    with Path(path).open("rb") as stream:
        return validate_evaluation_config(tomllib.load(stream))


def build_split_plan(inputs, targets, config):
    """No random splitting; store shipment IDs rather than unstable row positions."""
    config = validate_evaluation_config(config)
    aligned = align_targets(inputs, targets)
    ordered = inputs.sort_values(["as_of_utc", "shipment_id"], kind="stable").copy()
    ordered["label_available_utc"] = ordered.shipment_id.map(aligned.actual_delivery_utc)
    holdout_size = config["holdout_size"]
    if len(ordered) <= holdout_size:
        raise ValueError("Dataset is too small for the requested holdout")
    holdout = ordered.iloc[-holdout_size:]
    prior = ordered.iloc[:-holdout_size]
    cutoff = holdout.as_of_utc.iloc[0]
    if prior.as_of_utc.max() >= cutoff:
        raise ValueError("Holdout boundary splits simultaneous checkpoints")
    development = prior.loc[prior.label_available_utc < cutoff]
    excluded = prior.loc[prior.label_available_utc >= cutoff]
    first_validation_index = len(development) - config["n_validation_folds"] * config["validation_size"]
    if first_validation_index < config["minimum_training_size"]:
        raise ValueError("Insufficient development shipments for requested folds")
    folds = []
    for number in range(config["n_validation_folds"]):
        start = first_validation_index + number * config["validation_size"]
        end = start + config["validation_size"]
        validation = development.iloc[start:end]
        earlier = development.iloc[:start]
        boundary = validation.as_of_utc.iloc[0]
        if earlier.as_of_utc.max() >= boundary:
            raise ValueError("Validation boundary splits simultaneous checkpoints")
        training = earlier.loc[earlier.label_available_utc < boundary]
        unavailable = earlier.loc[earlier.label_available_utc >= boundary]
        if len(training) < config["minimum_training_size"]:
            raise ValueError(f"Fold {number + 1} has too few label-available training shipments")
        folds.append({
            "fold": number + 1, "cutoff_utc": boundary.isoformat(),
            "training_ids": training.shipment_id.tolist(),
            "validation_ids": validation.shipment_id.tolist(),
            "excluded_unavailable_label_ids": unavailable.shipment_id.tolist(),
        })
    return {
        "schema_version": 1, "evaluation_config": config,
        "holdout_cutoff_utc": cutoff.isoformat(),
        "development_ids": development.shipment_id.tolist(),
        "holdout_ids": holdout.shipment_id.tolist(),
        "excluded_unavailable_label_ids": excluded.shipment_id.tolist(),
        "folds": folds,
        "holdout_scored": False,
        "policy": "Strict checkpoint chronology; actual delivery must be strictly before training cutoff.",
    }
