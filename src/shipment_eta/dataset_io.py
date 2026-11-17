"""Audited CSV export with explicit input/target separation and provenance."""

from importlib.metadata import version
from pathlib import Path
import hashlib
import json
import platform
import tempfile

import pandas as pd

from .config import load_config
from .schema import (IDENTIFIERS, INPUT_TIMES, FUTURE_TIMES, TARGET_COLUMNS,
                     checkpoint_inputs, validate_inputs, validate_journeys)
from .simulation import GENERATOR_VERSION, generate_journeys


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_table(frame: pd.DataFrame, path: Path) -> None:
    exported = frame.copy(deep=True)
    for name in INPUT_TIMES + FUTURE_TIMES:
        if name in exported:
            exported[name] = exported[name].map(lambda value: value.isoformat())
    exported.to_csv(path, index=False, encoding="utf-8", float_format="%.17g", lineterminator="\n")


def read_table(path: str | Path) -> pd.DataFrame:
    """Read our CSV format without silently interpreting naive times as UTC."""
    frame = pd.read_csv(path, dtype={name: str for name in IDENTIFIERS})
    for name in INPUT_TIMES + FUTURE_TIMES:
        if name in frame:
            text = frame[name].astype("string")
            if not text.str.endswith(("+00:00", "Z"), na=False).all():
                raise ValueError(f"{name} must contain explicit UTC offsets")
            frame[name] = pd.to_datetime(text, format="ISO8601", utc=True, errors="raise")
    return frame


def _summary(series):
    return {"min": float(series.min()), "median": float(series.median()),
            "mean": float(series.mean()), "p95": float(series.quantile(0.95)),
            "max": float(series.max())}


def audit_journeys(frame: pd.DataFrame) -> dict:
    """Describe structural behavior without fitting or scoring any model."""
    deviation = frame.signed_arrival_deviation_h
    port_lateness = (frame.actual_port_arrival_utc - frame.scheduled_port_arrival_utc).dt.total_seconds() / 3600
    ordered = frame.sort_values(["as_of_utc", "shipment_id"], kind="stable")
    # This is split feasibility only; the actual split manifest is later work.
    holdout_n = max(1, min(60, len(frame) // 5))
    cutoff = ordered.iloc[-holdout_n].as_of_utc
    prior = ordered.iloc[:-holdout_n]
    eligible = prior.actual_delivery_utc < cutoff
    return {
        "schema_version": 1,
        "synthetic": True,
        "n_shipments": len(frame),
        "unique_shipment_ids": int(frame.shipment_id.nunique()),
        "missing_values": int(frame.isna().sum().sum()),
        "route_counts": {k: int(v) for k, v in sorted(frame.route_id.value_counts().items())},
        "arrival_counts": {"early": int((deviation < 0).sum()),
                           "exactly_on_time": int((deviation == 0).sum()),
                           "late": int((deviation > 0).sum())},
        "remaining_h": _summary(frame.remaining_h),
        "delay_magnitude_h": _summary(frame.delay_magnitude_h),
        "signed_arrival_deviation_h": _summary(deviation),
        "port_arrival_deviation_h": _summary(port_lateness),
        "dispatch_wait_h": _summary(frame.dispatch_wait_h),
        "late_port_but_not_late_delivery": int(((port_lateness > 0) & (deviation <= 0)).sum()),
        "late_port_and_amplified_final_lateness": int(((port_lateness > 0) & (deviation > port_lateness)).sum()),
        "disruptions": {name: int(frame[name].sum()) for name in
                        ["departure_disruption", "port_disruption", "inland_disruption"]},
        "split_feasibility": {"holdout_shipments": holdout_n,
                              "first_holdout_checkpoint_utc": cutoff.isoformat(),
                              "prior_shipments": len(prior),
                              "eligible_training_shipments": int(eligible.sum()),
                              "prior_shipments_with_unavailable_labels": int((~eligible).sum())},
        "interpretation": "Structural audit of synthetic assumptions; no predictive performance has been measured.",
    }


def _json(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def export_dataset(config_path: str | Path, output_dir: str | Path) -> dict:
    """Validate in a temporary directory and publish a new, complete directory.

    Existing output directories are refused, never overwritten. Dataset CSV
    bytes are reproducible in the same environment; manifests record environment
    differences and therefore need not have identical hashes across machines.
    """
    config_path, output_dir = Path(config_path).resolve(), Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output already exists: {output_dir}. Choose a new output directory.")
    config = load_config(config_path)
    journeys = generate_journeys(config)
    inputs = checkpoint_inputs(journeys)
    targets = journeys.loc[:, TARGET_COLUMNS].copy()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".shipment-build-", dir=output_dir.parent) as temporary:
        bundle = Path(temporary) / "bundle"
        bundle.mkdir()
        tables = {"journeys.csv": journeys, "checkpoint_inputs.csv": inputs, "targets.csv": targets}
        for name, table in tables.items():
            write_table(table, bundle / name)
            restored = read_table(bundle / name)
            pd.testing.assert_frame_equal(table.reset_index(drop=True), restored,
                                          check_dtype=False, check_exact=False, rtol=0, atol=1e-9)
        validate_journeys(read_table(bundle / "journeys.csv"), config)
        validate_inputs(read_table(bundle / "checkpoint_inputs.csv"))
        audit = audit_journeys(journeys)
        _json(bundle / "audit.json", audit)
        _json(bundle / "simulation_config.json", config)
        source_root = Path(__file__).resolve().parents[2]
        source_hashes = {p.relative_to(source_root).as_posix(): sha256(p)
                         for p in sorted(Path(__file__).parent.glob("*.py"))}
        manifest = {
            "generator_version": GENERATOR_VERSION,
            "schema_version": 1,
            "seed": config["seed"],
            "rng": "SeedSequence(seed).spawn(2)[0] -> Generator(PCG64(child)); observation child 1 reserved",
            "python": platform.python_version(), "platform": platform.platform(),
            "packages": {name: version(name) for name in ["numpy", "pandas"]},
            "config_file_sha256": sha256(config_path),
            "source_sha256": source_hashes,
            "files_sha256": {p.name: sha256(p) for p in sorted(bundle.iterdir())},
            "validation": "in-memory invariants and CSV round trip passed",
            "prediction_input_columns": list(inputs.columns),
            "target_columns": list(targets.columns),
        }
        _json(bundle / "manifest.json", manifest)
        if output_dir.exists():
            raise FileExistsError(f"Output appeared during generation: {output_dir}")
        bundle.rename(output_dir)
    return audit
