# Final technical and submission audit

Audit date: 16 November 2026. Scope: the delivered source, configurations, tests,
recorded evidence, documentation, report builder and local submission package.

**Result:** the reviewed workflow passes its tests, reproduces the recorded
experiment and retains the frozen implementation. The report states the negative
holdout comparison explicitly. No functional defect was found in the reviewed
scope; testing does not prove that a program is bug-free or production-ready.
GitHub publication, recruiter access and a published-checkout check remain open.

## Executed checks

| Area | Method | Observed result |
| --- | --- | --- |
| Dependencies | Compare installed metadata against all 23 pins; run pip check | Exact pins; no broken requirements |
| Application tests | Isolated review tree, pytest with warnings as errors | 186 passed in 81.64 s |
| Default workflow | Run all seven CLI entry points in sequence with new outputs | All completed successfully |
| Explicit input path | Predict from the generated 20-column example CSV | Completed; prediction/contribution/explanation files match the demo |
| Simulation replay | Compare all three generated CSV SHA-256 hashes | Exact match with original evaluation |
| Results replay | Compare final summary with original saved summary | Exact equality |
| Artifact integrity | Check each dataset/artifact bundle's recorded file hashes | All match |
| Frozen implementation | Compare source, CLI, configuration and evaluation-protocol bytes | All 29 frozen files unchanged |
| Independent calculations | Recompute from raw timestamps and saved predictions, without importing project metric/model functions | 432 recorded checks passed |
| Windows evidence | Read supplied clean-environment transcript and post-copy freeze screenshot | 186 tests passed; reproduction matched; frozen files unchanged |

The current Linux run used Python 3.12.14 and the previously clean-installed
isolated environment. It was a new source/output replay, not a second package
installation during this audit. The earlier clean-install procedure and the
candidate's Windows/Python 3.12.1 observations are recorded in
[ENVIRONMENT.md](../docs/ENVIRONMENT.md). No remote Windows execution is claimed.

Replays repeat the same exposed experiment. They are not additional independent
holdout evaluations and do not strengthen the statistical sample size.

## Code and numerical review

The review covered all application modules, seven CLI scripts, seven test files,
three TOML configurations and the evaluation protocol. Checks covered:

- Point-in-time input/target separation, UTC handling, finite values, IDs,
  chronological ordering and delivery-label availability.
- Simulator timing, target arithmetic and integer-nanosecond dispatch boundaries.
- Training-only scaling, fixed candidate selection, saved-model compatibility,
  duration clipping, delay flooring and explanation reconstruction.
- Baseline, candidate, fold, pooled-validation and final metrics; paired errors;
  bootstrap bounds; slice counts and errors; original-versus-replay equality.
- Output-directory protection, integrity failures and the tested CLI error paths.

The independent audit also checks all 300 dispatch calculations and all 60
holdout explanations. These per-row assertions are additional to the 432
recorded comparisons; 432 is not a new application-test count. Metric comparisons
use an absolute tolerance of 1e-9 hours; explanation-term checks use 1e-10 hours.
Exact byte/JSON comparisons are identified separately above.

Records are in [verification/pytest.log](verification/pytest.log),
[verification/workflow.json](verification/workflow.json),
[verification/numeric_verification.json](verification/numeric_verification.json)
and [verification/audit_summary.json](verification/audit_summary.json).
[verify_run.py](../report_tools/verify_run.py) reruns the numerical audit after
the seven default workflow commands. It does not retrain or select a model.

## Verified interpretation

| Holdout measure, 60 shipments | Selected Ridge | Carry-forward |
| --- | ---: | ---: |
| ETA MAE, hours | 7.123037 | 7.063309 |
| ETA RMSE, hours | 10.403546 | 10.102236 |
| Delay magnitude MAE, hours | 5.776577 | 5.685048 |
| P95 absolute ETA error, hours | 28.754722 | 21.941458 |

Carry-forward minus Ridge MAE is -0.059728 h; its approximate 95% paired
bootstrap interval is [-1.130719, 1.013604] h. Ridge therefore has no demonstrated
holdout advantage. The interval also does not establish equivalence. The frozen
Ridge model is retained; no post-holdout tuning, seed search or dataset redesign
was performed. All data are synthetic and assumptions are not calibrated to
real shipment operations.

## Report and documentation changes

The PDF was rebuilt as three A4 pages with white backgrounds, black text and
black table rules. It uses embedded Liberation Sans fonts, matching left/right
margins, one table grid system, consistent headings and controlled subscripts.
The document covers the prediction contract, simulator equations, propagation,
feature/model choice, evaluation, a measured explanation, results and limitations.

Numbers are populated from checksum-verified evidence exports. Simulator
parameters are checked against the saved configuration; equations were traced
to the implementation. All three pages were rendered for visual inspection.
Automated PDF checks cover page count, bounds, glyph coverage, monochrome drawing
operators, raster channels, table geometry and reference links. The PDF builder
uses invariant metadata; two builds with the recorded report environment matched
byte-for-byte. The constant PDF metadata timestamp is intentional, not an audit
date. See [report_tools/README.md](../report_tools/README.md).

One documentation discrepancy was corrected: the generator uses the first child
of `SeedSequence(seed).spawn(2)`, rather than passing the root seed directly to
PCG64. The simulation itself was already correct and was not changed. Formatting,
report-generation utilities and audit records are outside the frozen application.

The package adds `.gitattributes` to preserve LF text bytes in Git checkouts,
including Windows checkouts. This protects byte-based source integrity from
automatic line-ending conversion; it does not alter the evaluated source files.

## Submission boundary

The reviewed source package contains code, tests, configuration, the dependency
snapshot, documentation, evidence exports, the PDF and its reproducible builder.
It excludes virtual environments, caches, generated model/data directories,
recruitment screenshots, private transcripts and embedded ZIP archives. The
README regenerates the dataset and complete model/evaluation artifacts.

The package's member bytes, extraction, links and frozen-file hashes were checked.
Preserve the original local `data/generated` and `artifacts` directories when
applying the documentation update to the existing Windows project.

Still required before sending:

1. Candidate review of the displayed name, report and submission checklist.
2. Publish the reviewed repository and verify recruiter access.
3. Verify setup/tests/workflow from the actual published checkout in a new location.
4. Include the real repository URL and reviewed PDF in the submission.

No repository was published and no message was sent during this audit. Operational
accuracy, new routes, missing-data recovery, shared queues, reporting lag,
production load and calibrated individual ETA intervals remain unverified.
There is no justified guarantee of zero defects or a particular recruitment rank.
