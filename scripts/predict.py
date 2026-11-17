"""Predict final delivery ETA and delay with exact remaining-time explanations."""

from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shipment_eta.inference import run_predictions


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--demo", action="store_true", help="Use the first five saved training IDs; no performance evaluation")
    source.add_argument("--input", type=Path, help="CSV containing exactly the checkpoint input columns")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "artifacts/stage6")
    parser.add_argument("--split-manifest", type=Path, default=ROOT / "artifacts/stage5/split_manifest.json")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/generated", help="Used only with --demo")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    output = args.output_dir or ROOT / "artifacts" / ("stage7_demo" if args.demo else "predictions")
    try:
        summary = run_predictions(args.model_dir, args.split_manifest, output,
                                  input_path=args.input, demo_data_dir=args.data_dir if args.demo else None)
    except (ValueError, OSError, OverflowError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Predictions and explanations written to: {output.resolve()}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
