"""Checkpoint-only batch predictions, readable explanations and preserved holdout."""

from pathlib import Path
import csv
import json
import tempfile

import pandas as pd

from .dataset_io import read_table, sha256, write_table
from .explanations import predict_and_explain
from .schema import INPUT_COLUMNS, validate_inputs
from .training import load_trusted_model


def _json(path, content):
    path.write_text(json.dumps(content, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def read_prediction_inputs(path):
    """Reject duplicate headers before pandas can rename ambiguous columns."""
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream), [])
    if len(header) != len(set(header)) or set(header) != set(INPUT_COLUMNS):
        raise ValueError("Prediction CSV must have exactly the unique checkpoint input columns")
    frame = read_table(path)
    validate_inputs(frame)
    return frame


def _markdown_text(value):
    return str(value).replace("\\", "\\\\").replace("|", "\\|").replace("`", "\\`").replace("\n", " ").replace("\r", " ")


def render_explanation_markdown(document, mode):
    records = document["records"]
    lines = ["# Shipment predictions and explanations", "",
             f"Predictor: `{document['predictor']}`.", "",
             "Development demonstration on training records; no accuracy is measured." if mode == "development_demo" else
             "Checkpoint-only input predictions; no outcomes or performance metrics are used.", "",
             document["interpretation"], "", "Reference: " + document["reference"], "",
             f"Showing the first {min(5, len(records))} of {len(records)} shipments; JSON/CSV contain every input row.", ""]
    for record in records[:5]:
        lines.extend([f"## Shipment {_markdown_text(record['shipment_id'])}", "",
                      f"Checkpoint: {record['as_of_utc']}. Scheduled delivery: {record['scheduled_delivery_utc']}.", "",
                      f"**Predicted ETA: {record['predicted_eta_utc']}**", "",
                      f"Remaining: {record['predicted_remaining_h']:.3f} h. "
                      f"Signed deviation from schedule: {record['predicted_signed_deviation_h']:+.3f} h. "
                      f"Delay magnitude: {record['predicted_delay_magnitude_h']:.3f} h.", "",
                      "| Remaining-time calculation | Hours |", "| --- | ---: |",
                      f"| Reference prediction | {record['reference_remaining_h']:+.6f} |"])
        for group in record["group_contributions"]:
            lines.append(f"| {group['label']} | {group['contribution_h']:+.6f} |")
        lines.extend([f"| Raw predicted remaining time | {record['raw_remaining_h']:.6f} |",
                      f"| Nonnegative-duration adjustment | {record['remaining_adjustment_h']:+.6f} |",
                      f"| Served remaining time | {record['predicted_remaining_h']:.6f} |", "",
                      "Groups are ordered by absolute contribution. Positive adds hours relative to the reference; "
                      "negative subtracts hours. All individual terms are in contributions.csv and explanations.json.", "",
                      f"Signed deviation = deadline offset ({record['deadline_offset_h']:+.6f} h) + served remaining "
                      f"({record['predicted_remaining_h']:.6f} h) = {record['predicted_signed_deviation_h']:+.6f} h. "
                      "Delay magnitude is this signed value floored at zero.", ""])
        if record["remaining_was_clipped"]:
            lines.extend(["**Negative raw duration was clipped to zero. Treat this as a model limitation.**", ""])
        if record["training_range_flags"]:
            lines.extend(["**Outside a fitted training feature range:** " + ", ".join(
                _markdown_text(flag["feature"]) for flag in record["training_range_flags"]) + ".", ""])
        else:
            lines.extend(["No individual feature-range flag. This does not establish prediction reliability." if document["range_check_applicable"]
                          else "Training-feature range checks do not apply to this fixed rule.", ""])
    return "\n".join(lines)


def plot_explanation(record, path):
    """A static contribution plot; use Agg canvas without changing global backend."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    groups = record["group_contributions"]
    values = [group["contribution_h"] for group in groups]
    figure = Figure(figsize=(10, 5.4), layout="constrained")
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    axis.barh([group["label"] for group in groups], values,
              color=["#b44936" if value >= 0 else "#246b9b" for value in values])
    axis.invert_yaxis()
    axis.axvline(0, color="#333333", linewidth=0.8)
    maximum = max(1.0, max(abs(value) for value in values))
    axis.set_xlim(min(0, min(values)) - maximum * 0.25, max(0, max(values)) + maximum * 0.25)
    for position, value in enumerate(values):
        axis.text(value + maximum * (0.025 if value >= 0 else -0.025), position,
                  f"{value:+.3f} h", va="center", ha="left" if value >= 0 else "right", fontsize=10)
    axis.set_xlabel("Change from the stated remaining-time reference (hours)")
    axis.set_title(f"Shipment {record['shipment_id']}: model contribution breakdown\n"
                   f"Reference {record['reference_remaining_h']:.3f} h + contributions = "
                   f"raw remaining {record['raw_remaining_h']:.3f} h\n"
                   f"Duration adjustment {record['remaining_adjustment_h']:.3f} h; "
                   f"served remaining {record['predicted_remaining_h']:.3f} h", fontsize=11, pad=14)
    figure.supxlabel("Model arithmetic, not causal effects or separate physical stage durations.", fontsize=9)
    figure.savefig(path, dpi=150)


def run_predictions(model_dir, split_path, output_dir, *, input_path=None, demo_data_dir=None):
    """Predict from one explicit CSV or five preselected training examples."""
    if (input_path is None) == (demo_data_dir is None):
        raise ValueError("Choose exactly one of input_path or demo_data_dir")
    model_dir, split_path, output_dir = Path(model_dir), Path(split_path), Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output already exists: {output_dir}. Preserve it; choose a new output directory.")
    model, metadata = load_trusted_model(model_dir)
    if sha256(split_path) != metadata["split_manifest_sha256"]:
        raise ValueError("Split manifest does not match the saved model")
    plan = json.loads(split_path.read_text(encoding="utf-8"))
    if plan["development_ids"] != metadata["training_ids"]:
        raise ValueError("Saved training IDs differ from the frozen development partition")
    mode = "development_demo" if demo_data_dir is not None else "checkpoint_csv"
    source_path = Path(demo_data_dir) / "checkpoint_inputs.csv" if demo_data_dir is not None else Path(input_path)
    source_hash = sha256(source_path)
    if demo_data_dir is not None and source_hash != metadata["dataset_files_sha256"]["checkpoint_inputs.csv"]:
        raise ValueError("Demo checkpoint inputs differ from the training dataset")
    inputs = read_prediction_inputs(source_path)
    if demo_data_dir is not None:
        inputs = inputs.set_index("shipment_id").loc[metadata["training_ids"][:5]].reset_index()
    if set(inputs.shipment_id).intersection(plan["holdout_ids"]):
        raise ValueError("Reserved holdout shipment IDs are not allowed in this stage's prediction command")
    predictions, document = predict_and_explain(model, metadata, inputs)
    records = document["records"]
    summary = {
        "schema_version": 1, "mode": mode, "predictor": document["predictor"],
        "n_predictions": len(inputs), "shipment_ids": inputs.shipment_id.tolist(),
        "holdout_ids_predicted": 0, "holdout_scored": False, "performance_evaluated": False,
        "model_refitted": False,
        "remaining_clipped_count": int(predictions.was_clipped.sum()),
        "shipments_with_feature_range_flags": sum(bool(r["training_range_flags"]) for r in records),
        "max_explanation_reconstruction_error_h": max(r["reconstruction_abs_error_h"] for r in records),
        "interpretation": "Development training examples; explanation demonstration, not accuracy evaluation." if mode == "development_demo"
                          else "Checkpoint-only inference; no outcomes or accuracy evaluation.",
        "first_prediction": {key: records[0][key] for key in ["shipment_id", "predicted_eta_utc",
                             "predicted_remaining_h", "predicted_signed_deviation_h", "predicted_delay_magnitude_h"]},
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".prediction-build-", dir=output_dir.parent) as temporary:
        bundle = Path(temporary) / "bundle"
        bundle.mkdir()
        write_table(inputs, bundle / "checkpoint_inputs.csv")
        table = inputs.set_index("shipment_id")[["as_of_utc", "scheduled_delivery_utc"]].join(predictions).reset_index()
        for name in ["as_of_utc", "scheduled_delivery_utc", "predicted_eta_utc"]:
            table[name] = table[name].map(lambda value: value.isoformat())
        table.to_csv(bundle / "predictions.csv", index=False, float_format="%.17g", lineterminator="\n")
        flat_terms = [{"shipment_id": record["shipment_id"], **term} for record in records for term in record["feature_terms"]]
        pd.DataFrame(flat_terms).to_csv(bundle / "contributions.csv", index=False, float_format="%.17g", lineterminator="\n")
        _json(bundle / "explanations.json", document)
        _json(bundle / "summary.json", summary)
        (bundle / "explanations.md").write_text(render_explanation_markdown(document, mode), encoding="utf-8")
        plot_explanation(records[0], bundle / "explanation_example.png")
        source_root = Path(__file__).resolve().parents[2]
        _json(bundle / "run_manifest.json", {
            "schema_version": 1, "model_manifest_sha256": sha256(model_dir / "run_manifest.json"),
            "model_sha256": sha256(model_dir / "model.joblib"), "split_manifest_sha256": sha256(split_path),
            "input_source_sha256": source_hash, "input_mode": mode,
            "model_environment": metadata["environment"],
            "source_sha256": {p.relative_to(source_root).as_posix(): sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
            "files_sha256": {p.name: sha256(p) for p in sorted(bundle.iterdir())},
            "holdout_scored": False, "performance_evaluated": False,
        })
        if output_dir.exists():
            raise FileExistsError(f"Output appeared during prediction: {output_dir}")
        bundle.rename(output_dir)
    return summary
