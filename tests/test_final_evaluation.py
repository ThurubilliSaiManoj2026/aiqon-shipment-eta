"""Final-evaluation integration uses a separate seed, never the main holdout."""

from pathlib import Path
import json
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from shipment_eta.dataset_io import export_dataset, sha256
from shipment_eta.evaluation_freeze import FINAL_PROTOCOL, freeze_evaluation
from shipment_eta.final_evaluation import evaluate_holdout, final_metrics, paired_bootstrap
from shipment_eta.preparation import prepare_development
from shipment_eta.robustness import review_development
from shipment_eta.schema import INPUT_COLUMNS
from shipment_eta.training import train_comparison

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def evidence(tmp_path_factory):
    root = tmp_path_factory.mktemp("final-evaluation-separate-seed")
    config = root / "simulation_test.toml"
    text = (ROOT / "config/simulation.toml").read_text(encoding="utf-8")
    assert text.count("seed = 42") == 1
    config.write_text(text.replace("seed = 42", "seed = 314159"), encoding="utf-8")
    data, stage5, model, review = (root / name for name in ["data", "stage5", "model", "review"])
    export_dataset(config, data)
    prepare_development(data, ROOT / "config/evaluation.toml", stage5)
    train_comparison(data, stage5, ROOT / "config/training.toml", model)
    review_development(data, stage5, model, review)
    assert read_json(data / "manifest.json")["seed"] == 314159
    return data, stage5, model, review


@pytest.fixture(scope="module")
def frozen(evidence, tmp_path_factory):
    output = tmp_path_factory.mktemp("final-freeze") / "snapshot"
    freeze_evaluation(*evidence, output)
    return output


@pytest.fixture(scope="module")
def evaluated(evidence, frozen, tmp_path_factory):
    output = tmp_path_factory.mktemp("final-evaluation") / "results"
    summary = evaluate_holdout(*evidence, frozen, output)
    return output, summary


def test_freeze_creates_no_predictions_and_preserves_evidence(evidence, tmp_path, monkeypatch):
    from sklearn.pipeline import Pipeline
    before = {str(p): sha256(p) for folder in evidence for p in folder.iterdir()}

    def no_predict(*args, **kwargs):
        pytest.fail("Freeze attempted prediction")

    monkeypatch.setattr(Pipeline, "predict", no_predict)
    output = tmp_path / "freeze"
    summary = freeze_evaluation(*evidence, output)
    assert summary["holdout_scored"] is False
    assert summary["holdout_shipments"] == 60
    assert summary["bootstrap_resamples"] == 5000
    assert set(p.name for p in output.iterdir()) == {"freeze.json", "run_manifest.json"}
    snapshot = read_json(output / "freeze.json")
    plan = read_json(evidence[1] / "split_manifest.json")
    assert snapshot["holdout_ids"] == plan["holdout_ids"]
    assert snapshot["training_ids"] == plan["development_ids"]
    assert not set(snapshot["training_ids"]).intersection(snapshot["holdout_ids"])
    assert snapshot["protocol"] == FINAL_PROTOCOL
    assert "scripts/evaluate_holdout.py" in snapshot["executable_sha256"]
    assert "docs/FINAL_EVALUATION_PROTOCOL.md" in snapshot["executable_sha256"]
    assert before == {str(p): sha256(p) for folder in evidence for p in folder.iterdir()}
    with pytest.raises(FileExistsError):
        freeze_evaluation(*evidence, output)


def test_hand_calculated_metrics():
    errors = pd.DataFrame({"eta_error_h": [-3.0, 4.0], "delay_error_h": [-1.0, 2.0], "was_clipped": [False, True]})
    result = final_metrics(errors)
    assert result["eta_mae_h"] == 3.5
    assert result["eta_rmse_h"] == pytest.approx(np.sqrt(12.5))
    assert result["eta_bias_h"] == .5
    assert result["delay_mae_h"] == 1.5
    assert result["eta_p95_abs_error_h"] == pytest.approx(3.95)
    assert result["eta_max_abs_error_h"] == 4


def test_bootstrap_constant_errors_have_exact_intervals():
    selected = pd.Series([1.0, -1.0, 1.0], index=["a", "b", "c"])
    baseline = pd.Series([-2.0, 2.0, 2.0], index=["c", "a", "b"])
    result = paired_bootstrap(selected, baseline)
    expected = {"estimate_h": 1.0, "lower_h": 1.0, "upper_h": 1.0}
    assert result["selected_eta_mae_h"] == expected
    assert result["paired_eta_mae_improvement_h"] == expected


def test_bootstrap_is_paired_reproducible_and_id_aligned():
    selected = pd.Series([1.0, -3.0, 9.0, -4.0], index=list("abcd"))
    # Each matched absolute baseline error is exactly two hours larger.
    baseline = pd.Series([11.0, 3.0, 6.0, 5.0], index=list("cadb"))
    first = paired_bootstrap(selected, baseline)
    assert first == paired_bootstrap(selected, baseline.iloc[::-1])
    assert first["paired_eta_mae_improvement_h"] == {"estimate_h": 2.0, "lower_h": 2.0, "upper_h": 2.0}
    assert first["selected_eta_mae_h"]["lower_h"] < first["selected_eta_mae_h"]["upper_h"]


@pytest.mark.parametrize("bad", [pd.Series([], dtype=float), pd.Series([1.0]),
                                 pd.Series([1.0, np.nan]), pd.Series([1.0, np.inf]),
                                 pd.Series([True, False]), pd.Series([1.0, 2.0], index=["a", "a"])])
def test_invalid_bootstrap_inputs_rejected(bad):
    with pytest.raises(ValueError):
        paired_bootstrap(bad, bad)


@pytest.mark.parametrize("change", ["predictor", "holdout_id", "protocol", "executable_hash", "raw_bytes"])
def test_modified_freeze_is_rejected_before_prediction(evidence, frozen, tmp_path, monkeypatch, change):
    from shipment_eta import final_evaluation
    copied = tmp_path / "freeze"
    shutil.copytree(frozen, copied)
    path = copied / "freeze.json"
    content = read_json(path)
    if change == "predictor":
        content["predictor"] = "scheduled"
    elif change == "holdout_id":
        content["holdout_ids"][0] = content["training_ids"][0]
    elif change == "protocol":
        content["protocol"]["bootstrap_seed"] = 0
    elif change == "executable_hash":
        content["executable_sha256"]["scripts/evaluate_holdout.py"] = "0" * 64
    path.write_text(json.dumps(content) + (" " if change == "raw_bytes" else ""), encoding="utf-8")
    if change != "raw_bytes":
        manifest = read_json(copied / "run_manifest.json")
        manifest["files_sha256"]["freeze.json"] = sha256(path)
        (copied / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def no_prediction(*args):
        pytest.fail("Modified freeze reached final prediction")

    monkeypatch.setattr(final_evaluation, "predict_and_explain", no_prediction)
    with pytest.raises(ValueError):
        evaluate_holdout(*evidence, copied, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()


def test_missing_freeze_or_changed_review_stops_evaluation(evidence, frozen, tmp_path):
    with pytest.raises(FileNotFoundError):
        evaluate_holdout(*evidence, tmp_path / "missing", tmp_path / "output")
    copied = tmp_path / "review"
    shutil.copytree(evidence[3], copied)
    path = copied / "summary.json"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="checksum"):
        evaluate_holdout(*evidence[:3], copied, frozen, tmp_path / "output")


def test_final_outputs_use_only_frozen_holdout_and_metrics_recompute(evaluated, frozen):
    output, summary = evaluated
    snapshot = read_json(frozen / "freeze.json")
    predictions = pd.read_csv(output / "holdout_predictions.csv")
    metrics = pd.read_csv(output / "metrics.csv")
    assert summary["holdout_scored"] is True
    assert summary["model_refitted"] is summary["predictor_changed"] is False
    assert summary["predictor"] == snapshot["predictor"]
    assert summary["paired_vs_validation_baseline"]["baseline"] == snapshot["comparison_baseline"]
    for name, rows in predictions.groupby("predictor", sort=False):
        assert rows.shipment_id.tolist() == snapshot["holdout_ids"]
        assert rows.shipment_id.is_unique
        assert not set(rows.shipment_id).intersection(snapshot["training_ids"])
        score = metrics.loc[metrics.predictor == name].iloc[0]
        assert score.n == 60
        predicted_eta = pd.to_datetime(rows.predicted_eta_utc, format="ISO8601", utc=True)
        actual_eta = pd.to_datetime(rows.actual_delivery_utc, format="ISO8601", utc=True)
        timestamp_errors = (predicted_eta - actual_eta).dt.total_seconds() / 3600
        np.testing.assert_allclose(timestamp_errors, rows.eta_error_h, atol=1e-9)
        assert score.eta_mae_h == pytest.approx(timestamp_errors.abs().mean())
        assert score.eta_rmse_h == pytest.approx(np.sqrt((timestamp_errors ** 2).mean()))
        assert score.delay_mae_h == pytest.approx((rows.predicted_delay_magnitude_h - rows.actual_delay_magnitude_h).abs().mean())
        assert score.eta_p95_abs_error_h == pytest.approx(timestamp_errors.abs().quantile(.95))
    explanations = read_json(output / "holdout_explanations.json")
    assert [r["shipment_id"] for r in explanations["records"]] == snapshot["holdout_ids"]
    for row in explanations["records"]:
        assert row["raw_remaining_h"] == pytest.approx(row["reference_remaining_h"] + sum(t["contribution_h"] for t in row["feature_terms"]))
    manifest = read_json(output / "run_manifest.json")
    assert manifest["freeze_sha256"] == sha256(frozen / "freeze.json")
    for name, checksum in manifest["files_sha256"].items():
        assert sha256(output / name) == checksum
    assert (output / "holdout_results.png").read_bytes().startswith(b"\x89PNG")
    assert "holdout is now exposed" in (output / "evaluation.md").read_text(encoding="utf-8")


def test_evaluation_never_fits_and_supplies_only_holdout_checkpoint_inputs(evidence, frozen, evaluated, tmp_path, monkeypatch):
    from sklearn.pipeline import Pipeline
    before = {str(p): sha256(p) for folder in [*evidence, frozen] for p in folder.iterdir()}
    snapshot = read_json(frozen / "freeze.json")
    original_predict = Pipeline.predict
    observed = []

    def no_fit(*args, **kwargs):
        pytest.fail("Final evaluation attempted fitting")

    def observed_predict(self, inputs, *args, **kwargs):
        assert set(inputs.columns) == set(INPUT_COLUMNS)
        assert inputs.shipment_id.tolist() == snapshot["holdout_ids"]
        observed.extend(inputs.shipment_id.tolist())
        return original_predict(self, inputs, *args, **kwargs)

    monkeypatch.setattr(Pipeline, "fit", no_fit)
    monkeypatch.setattr(Pipeline, "predict", observed_predict)
    output = tmp_path / "reproduction"
    summary = evaluate_holdout(*evidence, frozen, output)
    # This separate fixture may select a baseline, in which case no Ridge prediction is needed.
    assert not observed or observed == snapshot["holdout_ids"]
    assert summary == evaluated[1]
    for name in ["metrics.csv", "holdout_predictions.csv", "bootstrap_intervals.json"]:
        assert (output / name).read_bytes() == (evaluated[0] / name).read_bytes()
    assert before == {str(p): sha256(p) for folder in [*evidence, frozen] for p in folder.iterdir()}
    with pytest.raises(FileExistsError):
        evaluate_holdout(*evidence, frozen, output)


def test_final_cli_workflow_from_another_directory_and_overwrite_rejection(evidence, tmp_path):
    shared = ["--data-dir", str(evidence[0]), "--stage5-dir", str(evidence[1]),
              "--model-dir", str(evidence[2]), "--review-dir", str(evidence[3])]
    freeze = tmp_path / "freeze"
    first = subprocess.run([sys.executable, "-W", "error", str(ROOT / "scripts/freeze_evaluation.py"),
                            *shared, "--output-dir", str(freeze)], cwd=tmp_path, text=True, capture_output=True)
    assert first.returncode == 0, first.stderr
    assert '"holdout_scored": false' in first.stdout
    command = [sys.executable, "-W", "error", str(ROOT / "scripts/evaluate_holdout.py"),
               *shared, "--freeze-dir", str(freeze), "--output-dir", str(tmp_path / "final")]
    final = subprocess.run(command, cwd=tmp_path, text=True, capture_output=True)
    assert final.returncode == 0, final.stderr
    assert '"holdout_scored": true' in final.stdout
    repeated = subprocess.run(command, cwd=tmp_path, text=True, capture_output=True)
    assert repeated.returncode == 1
    assert "Output already exists" in repeated.stderr
