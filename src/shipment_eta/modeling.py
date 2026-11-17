"""Interpretable remaining-time models with training-only standardization."""

from pathlib import Path
import math
import sys
import tomllib
from importlib.metadata import version

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .evaluation import align_targets, prediction_from_remaining
from .features import FEATURE_SETS, build_features

TRAINING_KEYS = {"schema_version", "feature_sets", "ridge_alphas", "ridge_solver",
                 "selection_mae_tolerance_h"}
MODEL_PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn", "joblib")


def validate_training_config(config):
    if not isinstance(config, dict) or set(config) != TRAINING_KEYS:
        raise ValueError("Unexpected training configuration keys")
    if type(config["schema_version"]) is not int or config["schema_version"] != 1:
        raise ValueError("Only training schema_version=1 is supported")
    groups = config["feature_sets"]
    if not isinstance(groups, list) or not groups or any(not isinstance(g, str) or g not in FEATURE_SETS for g in groups):
        raise ValueError("Unknown or empty feature_sets")
    if len(set(groups)) != len(groups):
        raise ValueError("Duplicate feature sets")
    alphas = config["ridge_alphas"]
    if not isinstance(alphas, list) or not alphas:
        raise ValueError("ridge_alphas must be a nonempty list")
    for alpha in alphas:
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("Ridge alphas must be finite and positive")
    if len(set(alphas)) != len(alphas):
        raise ValueError("Duplicate Ridge alphas")
    if config["ridge_solver"] != "svd":
        raise ValueError("This protocol requires the SVD Ridge solver")
    tolerance = config["selection_mae_tolerance_h"]
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Selection tolerance must be finite and nonnegative")
    return {**config, "feature_sets": list(groups), "ridge_alphas": [float(a) for a in alphas]}


def load_training_config(path):
    with Path(path).open("rb") as stream:
        return validate_training_config(tomllib.load(stream))


def candidate_id(feature_set, alpha):
    # repr preserves distinct finite float settings rather than rounding IDs.
    return f"ridge_{feature_set}_alpha_{repr(float(alpha)).removesuffix('.0')}"


def fit_candidate(inputs, targets, feature_set, alpha):
    if feature_set not in FEATURE_SETS:
        raise ValueError("Unknown feature set")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("Ridge alpha must be finite and positive")
    aligned = align_targets(inputs, targets)
    pipeline = Pipeline([
        ("features", FunctionTransformer(build_features, validate=False,
                                         kw_args={"feature_set": feature_set})),
        ("scale", StandardScaler()),
        ("regressor", Ridge(alpha=float(alpha), solver="svd")),
    ])
    pipeline.fit(inputs, aligned.remaining_h.to_numpy(dtype=float))
    return pipeline


def predict_candidate(pipeline, inputs):
    # Pipeline's feature step validates the complete checkpoint schema.
    raw = pd.Series(pipeline.predict(inputs), index=pd.Index(inputs.shipment_id, name="shipment_id"))
    return prediction_from_remaining(inputs, raw)


def select_candidate(pooled: pd.DataFrame, tolerance: float) -> dict:
    """Resolve near ties using fewer features, then stronger regularization."""
    required = {"candidate", "feature_set", "alpha", "eta_mae_h"}
    if pooled.empty or not required.issubset(pooled.columns) or not pooled.candidate.is_unique:
        raise ValueError("Invalid candidate comparison table")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Invalid selection tolerance")
    if not np.isfinite(pooled[["alpha", "eta_mae_h"]].to_numpy(dtype=float)).all():
        raise ValueError("Candidate scores must be finite")
    if (pooled.eta_mae_h < 0).any() or (pooled.alpha <= 0).any():
        raise ValueError("Candidate MAE must be nonnegative and alpha positive")
    if not pooled.feature_set.isin(FEATURE_SETS).all():
        raise ValueError("Unknown candidate feature set")
    minimum = float(pooled.eta_mae_h.min())
    eligible = pooled.loc[pooled.eta_mae_h <= minimum + tolerance].copy()
    eligible["n_features"] = eligible.feature_set.map(lambda group: len(FEATURE_SETS[group]))
    chosen = eligible.sort_values(["n_features", "alpha", "candidate"],
                                  ascending=[True, False, True], kind="stable").iloc[0]
    return {"candidate": chosen.candidate, "feature_set": chosen.feature_set,
            "alpha": float(chosen.alpha), "validation_eta_mae_h": float(chosen.eta_mae_h),
            "absolute_minimum_eta_mae_h": minimum,
            "gap_to_absolute_minimum_h": float(chosen.eta_mae_h - minimum),
            "selection_mae_tolerance_h": float(tolerance)}


def coefficient_table(pipeline, feature_set):
    scaler = pipeline.named_steps["scale"]
    ridge = pipeline.named_steps["regressor"]
    names = FEATURE_SETS[feature_set]
    if list(scaler.feature_names_in_) != names:
        raise ValueError("Fitted scaler feature order differs from feature schema")
    return pd.DataFrame({"feature": names, "training_mean": scaler.mean_,
                         "training_scale": scaler.scale_,
                         "coefficient_per_training_std_h": ridge.coef_,
                         "coefficient_per_raw_unit_h": ridge.coef_ / scaler.scale_})


def current_model_environment():
    return {"python_major_minor": list(sys.version_info[:2]),
            "packages": {package: version(package) for package in MODEL_PACKAGES}}
