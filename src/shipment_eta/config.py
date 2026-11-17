"""Strict validation of the explicit, synthetic simulation assumptions."""

from copy import deepcopy
from pathlib import Path
import math
import tomllib

import pandas as pd


class ConfigurationError(ValueError):
    """A simulation parameter is missing, unknown, or outside its domain."""


SECTION_KEYS = {
    "departure": {"normal_mean_h", "normal_std_h", "early_bound_h",
                  "disruption_probability", "disruption_min_h", "disruption_max_h"},
    "voyage": {"weather_beta_a", "weather_beta_b", "weather_effect_h",
               "noise_std_fraction", "minimum_duration_fraction"},
    "port": {"congestion_beta_a", "congestion_beta_b", "base_multiplier",
             "congestion_multiplier", "noise_std_h", "minimum_duration_h",
             "disruption_probability", "disruption_min_h", "disruption_max_h"},
    "inland": {"noise_std_fraction", "minimum_duration_h",
               "disruption_probability", "disruption_min_h", "disruption_max_h"},
}
ROUTE_TEXT = {"route_id", "origin", "destination_port", "destination"}
ROUTE_NUMERIC = {"voyage_h", "port_processing_h", "inland_h",
                 "delivery_buffer_h", "dispatch_interval_h"}
TOP_KEYS = {"schema_version", "seed", "n_shipments", "scheduled_start_utc",
            "departure_spacing_h", "route_assignment", "routes", *SECTION_KEYS}


def _keys(value, expected, label):
    if not isinstance(value, dict):
        raise ConfigurationError(f"{label} must be a table")
    missing, unknown = expected - value.keys(), value.keys() - expected
    if missing or unknown:
        raise ConfigurationError(f"{label}: missing={sorted(missing)}, unknown={sorted(unknown)}")


def _number(value, label, *, minimum=None, strict=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ConfigurationError(f"{label} must be a finite number")
    if minimum is not None and (value <= minimum if strict else value < minimum):
        relation = ">" if strict else ">="
        raise ConfigurationError(f"{label} must be {relation} {minimum}")


def validate_config(config: dict) -> dict:
    """Return an independent validated copy, rejecting misspelled parameters."""
    _keys(config, TOP_KEYS, "config")
    for name in ("schema_version", "seed", "n_shipments"):
        if type(config[name]) is not int:
            raise ConfigurationError(f"{name} must be an integer")
    if config["schema_version"] != 1:
        raise ConfigurationError("Only schema_version=1 is supported")
    if config["seed"] < 0 or config["n_shipments"] <= 0:
        raise ConfigurationError("seed must be nonnegative and n_shipments positive")
    if config["route_assignment"] != "balanced_then_shuffled":
        raise ConfigurationError("route_assignment must be balanced_then_shuffled")
    _number(config["departure_spacing_h"], "departure_spacing_h", minimum=0, strict=True)
    if not isinstance(config["scheduled_start_utc"], str):
        raise ConfigurationError("scheduled_start_utc must be a quoted ISO timestamp")
    try:
        start = pd.Timestamp(config["scheduled_start_utc"])
        if pd.isna(start) or start.tzinfo is None or start.utcoffset().total_seconds() != 0:
            raise ValueError("not an aware UTC timestamp")
        start.as_unit("ns")
        start + pd.Timedelta(hours=config["departure_spacing_h"] * (config["n_shipments"] - 1))
    except (ValueError, TypeError, OverflowError) as exc:
        raise ConfigurationError("scheduled_start_utc/schedule must be representable in UTC") from exc

    for section, keys in SECTION_KEYS.items():
        values = config[section]
        _keys(values, keys, section)
        for name, value in values.items():
            _number(value, f"{section}.{name}")
            if name not in {"normal_mean_h", "early_bound_h"}:
                _number(value, f"{section}.{name}", minimum=0)
            if name.endswith(("beta_a", "beta_b")) or name in {"minimum_duration_h", "minimum_duration_fraction"}:
                _number(value, f"{section}.{name}", minimum=0, strict=True)
        if "disruption_probability" in values:
            if values["disruption_probability"] > 1:
                raise ConfigurationError(f"{section}.disruption_probability must be in [0,1]")
            if values["disruption_min_h"] > values["disruption_max_h"]:
                raise ConfigurationError(f"{section}: disruption_min_h exceeds disruption_max_h")
    if config["voyage"]["minimum_duration_fraction"] > 1:
        raise ConfigurationError("voyage.minimum_duration_fraction must be <= 1")
    if config["departure"]["early_bound_h"] > 0:
        raise ConfigurationError("departure.early_bound_h must be <= 0")

    routes = config["routes"]
    if not isinstance(routes, list) or not routes:
        raise ConfigurationError("routes must be a nonempty array of tables")
    if config["n_shipments"] % len(routes):
        raise ConfigurationError("n_shipments must be divisible by the number of routes")
    seen = set()
    for index, route in enumerate(routes):
        _keys(route, ROUTE_TEXT | ROUTE_NUMERIC, f"routes[{index}]")
        for name in ROUTE_TEXT:
            if not isinstance(route[name], str) or not route[name].strip():
                raise ConfigurationError(f"routes[{index}].{name} must be nonempty text")
        if route["route_id"] in seen:
            raise ConfigurationError("route_id values must be unique")
        seen.add(route["route_id"])
        for name in ROUTE_NUMERIC:
            _number(route[name], f"routes[{index}].{name}", minimum=0,
                    strict=name != "delivery_buffer_h")
            try:
                interval = pd.Timedelta(hours=route[name]).value
                if name != "delivery_buffer_h" and interval <= 0:
                    raise ValueError("below timestamp resolution")
            except (ValueError, OverflowError) as exc:
                raise ConfigurationError(f"routes[{index}].{name} is not representable") from exc
    return deepcopy(config)


def load_config(path: str | Path) -> dict:
    try:
        with Path(path).open("rb") as stream:
            raw = tomllib.load(stream)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigurationError(f"Invalid TOML: {exc}") from exc
    return validate_config(raw)
