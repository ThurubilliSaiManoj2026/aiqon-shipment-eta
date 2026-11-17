"""Run from the project root: python scripts/generate_data.py."""

from pathlib import Path
import argparse
import json
import sys

# Keep the beginner setup usable without requiring an editable installation.
# Reusable logic lives in src/shipment_eta, not in this entry point.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shipment_eta.dataset_io import export_dataset


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate and validate synthetic shipment journeys.")
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "simulation.toml")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "generated")
    args = parser.parse_args(argv)
    try:
        audit = export_dataset(args.config, args.output_dir)
    except (ValueError, OSError, OverflowError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Validated dataset written to: {args.output_dir.resolve()}")
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
