# Measured evidence and provenance

The numerical report is based on the original local Linux frozen evaluation.
The files below are static, synthetic evidence; they are not production shipment
records, a newly tuned model, or a download of the candidate's Windows artifacts.

## What the exports contain

| Export | Original local path | Transformation |
| --- | --- | --- |
| [validation_baselines.csv](evidence/validation_baselines.csv) | `artifacts/stage5_verified/baseline_metrics.csv` | Byte-for-byte copy of validation baseline metrics |
| [validation_summary.json](evidence/validation_summary.json) | `artifacts/stage6_verified/summary.json` | Byte-for-byte copy of model-selection summary |
| [validation_candidates.csv](evidence/validation_candidates.csv) | `artifacts/stage6_verified/candidate_metrics.csv` | Byte-for-byte copy of fold and pooled candidate metrics |
| [development_review.json](evidence/development_review.json) | `artifacts/stage8_verified_final/summary.json` | Byte-for-byte copy of development review |
| [holdout_summary.json](evidence/holdout_summary.json) | `artifacts/stage9_final_verified/summary.json` | Byte-for-byte copy of final summary |
| [holdout_metrics.csv](evidence/holdout_metrics.csv) | `artifacts/stage9_final_verified/metrics.csv` | Byte-for-byte copy of final metrics |
| [holdout_predictions.csv](evidence/holdout_predictions.csv) | `artifacts/stage9_final_verified/holdout_predictions.csv` | Byte-for-byte copy of all 180 predictor/shipment rows |
| [holdout_slices.csv](evidence/holdout_slices.csv) | `artifacts/stage9_final_verified/slice_metrics.csv` | Byte-for-byte copy of descriptive slices |
| [first_holdout_explanation.json](evidence/first_holdout_explanation.json) | `artifacts/stage9_final_verified/holdout_explanations.json` | First-in-time holdout record SHP-000241 extracted without value changes |
| [dataset_audit.json](evidence/dataset_audit.json) | Review replay: `data/generated/audit.json` | Byte-for-byte copy from the unchanged simulator replay |
| [simulation_config.json](evidence/simulation_config.json) | Review replay: `data/generated/simulation_config.json` | Byte-for-byte saved configuration copy |
| [split_manifest.json](evidence/split_manifest.json) | Review replay: `artifacts/stage5/split_manifest.json` | Byte-for-byte split/availability record copy |

The `_verified` directories identify local runs made during staged development.
The Windows/default CLI directories have names such as `artifacts/stage9_final`.
These path differences do not imply different experiment settings.

The report's first explanation uses the earliest held-out checkpoint, not a
record chosen for low error. Feature/group values are unchanged; values displayed
in the PDF are rounded. Candidate metrics include fold and pooled rows; compare
pooled rows for model selection. Validation results were used for selection;
final holdout results were not.

The final audit added the last three exports for direct simulator and split
checks. The original nine exports and frozen experiment remain unchanged.
The new replay reproduced the original dataset hashes and final summary.
Current test, workflow and independent calculation records are in `verification`;
see [FINAL_AUDIT.md](FINAL_AUDIT.md).

## Verification boundaries

The supplied Windows transcript was read completely. Its printed final summary
was parsed and matched this local `holdout_summary.json` exactly. The transcript
also reports three dataset SHA-256 matches and exact original/replay summary
equality. Original and fresh Windows test runs passed all 186 tests with warnings
as errors. These are user-supplied observations; every Windows binary artifact
was not independently inspected. See [ENVIRONMENT.md](../docs/ENVIRONMENT.md).

A fresh Linux source tree and isolated environment separately completed all
186 tests and seven CLI stages. CSV bytes, split IDs, selection and final summary
matched the original local evaluation. The replay retained all 54 protected
original data/artifact files unchanged. This verifies reproduction of the same
exposed experiment, not another independent holdout assessment.

## Integrity and interpretation

[evidence_manifest.json](evidence/evidence_manifest.json) records each exported
file's SHA-256, source-file hash, transformation, the original local freeze/model
hashes, dataset CSV hashes and the frozen executable-file hashes. The local
original freeze SHA-256 is `2ce6ebd9ede5b3e452cecb1155a49f9209b554d9bc7aac4d67dc2bb53bd1290f`.
The frozen source/config/protocol bytes remain unchanged in the documentation
update. Hashes detect differences relative to the recorded bytes; they are not
external authentication or independent proof of execution order.

Binary model and provenance hashes can differ across environments. Recreate
models on the target machine using the documented versions; do not compare a
Windows model binary against the local Linux model hash as a portability test.

The main result is unchanged: Ridge ETA MAE is 7.123 h versus carry-forward's
7.063 h on 60 held-out synthetic shipments. The paired improvement interval
[-1.131, 1.014] h does not establish superiority or equivalence. Synthetic
assumptions and small diagnostic slices limit external conclusions. The report
includes these results without post-holdout model or dataset changes.
