"""Freeze the model, data, holdout IDs, source and final evaluation protocol."""

from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shipment_eta.evaluation_freeze import freeze_evaluation


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/generated")
    parser.add_argument("--stage5-dir", type=Path, default=ROOT / "artifacts/stage5")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "artifacts/stage6")
    parser.add_argument("--review-dir", type=Path, default=ROOT / "artifacts/stage8")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/stage9_freeze")
    args = parser.parse_args(argv)
    try:
        summary = freeze_evaluation(args.data_dir, args.stage5_dir, args.model_dir, args.review_dir, args.output_dir)
    except (ValueError, OSError, OverflowError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Evaluation freeze written to: {args.output_dir.resolve()}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
