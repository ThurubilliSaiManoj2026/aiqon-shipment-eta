"""Freeze time-based splits, build features and evaluate validation baselines."""

from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shipment_eta.preparation import prepare_development


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/generated")
    parser.add_argument("--config", type=Path, default=ROOT / "config/evaluation.toml")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/stage5")
    args = parser.parse_args(argv)
    try:
        summary = prepare_development(args.data_dir, args.config, args.output_dir)
    except (ValueError, OSError, OverflowError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Development preparation written to: {args.output_dir.resolve()}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
