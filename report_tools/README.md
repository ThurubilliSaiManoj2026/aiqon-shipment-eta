# Rebuild the technical report

The checked PDF is included in `reports/Shipment_ETA_Technical_Report.pdf`.
Regenerating it is optional and does not run training or evaluation. The builder
checks the SHA-256 hashes of the static exports in `reports/evidence` and uses
their measured values directly. It also checks the simulator parameters shown
in the compact equation table. Do not edit evidence to improve results.

## Optional report environment

Use a separate environment so report dependencies do not change the tested
application environment. From the repository root in Windows PowerShell:

```powershell
py -3.12 -m venv artifacts/report_env
.\artifacts\report_env\Scripts\python.exe -m pip install -r report_tools/requirements.txt
.\artifacts\report_env\Scripts\python.exe report_tools/build_report.py --output artifacts/Shipment_ETA_Technical_Report_rebuilt.pdf
```

Use a new environment destination if `artifacts/report_env` already exists.
On Linux, use `python3.12` and `artifacts/report_env/bin/python`.
The script writes the readable companion `reports/technical_report.md` too.
Its default PDF destination is `reports/Shipment_ETA_Technical_Report.pdf`;
`--output` above preserves that included copy.

Report generation was executed on Linux/Python 3.12.14 with ReportLab 4.4.9,
Pillow 13.0.0 and charset-normalizer 3.5.2. Windows report generation has not
been executed; the application's Windows reproduction is a separate check.
No LaTeX, browser, remote fonts or external rendering service is required.

Liberation Sans regular/bold/italic fonts are bundled with their original
license in `fonts/LICENSE.txt`. Embedded fonts and invariant PDF metadata
stabilize output. Two local builds matched byte-for-byte. The metadata date is
fixed by the renderer; use the final audit record for the verification date.
Across a different dependency or renderer version, recheck page count and layout
rather than assuming identical bytes.

## Independently check a completed default run

After executing all seven README workflow commands in a fresh checkout:

```powershell
.\.venv\Scripts\python.exe -W error report_tools/verify_run.py
```

This uses the application environment's NumPy/pandas, without importing project
metric/model implementations. It verifies preserved evidence, dataset and frozen
source hashes, package versions, timing/targets, splits, prediction metrics,
paired bootstrap intervals, explanations and descriptive slices. It writes
`artifacts/numerical_verification.json`, refusing to overwrite an existing file.
Choose a new `--output` path for a deliberate repeat. The default project root
is the repository containing the script; `--project-root` supports another
complete default-layout checkout.

This utility audits the fixed assignment experiment. It intentionally expects
the original 300 shipments and protocol. A revised experiment would require a
separate, newly defined evaluation; changing expected numbers is not validation.

The retained original audit has 432 recorded comparisons plus per-row assertions.
It is distinct from the 186 application tests. These utilities do not change the
frozen predictor, features, dataset generator or evaluation protocol.
