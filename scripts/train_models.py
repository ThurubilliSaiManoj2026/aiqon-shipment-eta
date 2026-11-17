"""Compare Ridge candidates on frozen folds and save the selected pipeline."""

from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shipment_eta.training import train_comparison


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/generated")
    parser.add_argument("--stage5-dir", type=Path, default=ROOT / "artifacts/stage5")
    parser.add_argument("--config", type=Path, default=ROOT / "config/training.toml")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/stage6")
    args = parser.parse_args(argv)
    try:
        summary = train_comparison(args.data_dir, args.stage5_dir, args.config, args.output_dir)
    except (ValueError, OSError, OverflowError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Model comparison written to: {args.output_dir.resolve()}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
