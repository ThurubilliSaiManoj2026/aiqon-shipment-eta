# Verified environments and evidence boundaries

## User-supplied Windows observations

Windows, Cursor, PowerShell 5.1; Python 3.12.1, 64-bit; Git 2.54.0.windows.1.
Original environment: `.venv`, pip 26.3. Fresh environment:
`artifacts/stage10_env`, pip 23.2.1. Both use the recorded dependency versions.

| Check | Supplied result |
| --- | --- |
| Original stage-9 tests | 186 passed in 55.93 s, warnings as errors |
| Fresh installation | All 23 recorded packages installed |
| Fresh pip check | No broken requirements found |
| Fresh tests | 186 passed in 68.56 s, warnings as errors |
| Seven CLI commands in separate directories | Completed |
| Three generated dataset CSVs | All SHA-256 comparisons MATCH |
| Reproduced versus original final summary | Exact equality: True |

The complete transcript was read and its final JSON compared programmatically
with the recorded local summary; they matched. These are supplied Windows
observations, not remote execution by the assistant or inspection of every
Windows artifact. Binary model hashes need not match across operating systems.

## Package versions

The authoritative project file is the existing `requirements.txt`, which this
documentation update does not replace. Both reported Windows installations used:

```text
cloudpickle==3.1.2
colorama==0.4.6
contourpy==1.4.0
cycler==0.12.1
fonttools==4.68.0
iniconfig==2.3.1
joblib==1.6.0
kiwisolver==1.5.1
matplotlib==3.11.2
narwhals==2.28.0
numpy==2.5.4
packaging==26.3
pandas==3.1.1
pillow==13.0.0
pluggy==1.7.0
Pygments==2.21.0
pyparsing==3.3.3
pytest==9.2.0
python-dateutil==4.0.0
scikit-learn==1.9.1
scipy==1.18.1
threadpoolctl==3.8.0
tzdata==2026.6
```

## Local clean reproduction

Linux/Python 3.12.14, pip 25.0.1; isolated environment with system site packages
disabled. The same 23 pins, transcribed from the supplied snapshot, were freshly
installed and checked via package metadata. `pip check` passed. A separate
source tree reconstructed from the delivered ZIPs had no earlier data or model.
**186 tests passed in 72.67 s**, warnings as errors; all seven CLI stages completed.

Generated CSV bytes, split/fold IDs and model selection matched the original
local evidence. Final summary equality was exact; maximum numeric final-metric
difference was 0.0. A separate-output command check also confirmed all 54 original
data/artifact files unchanged. These reproduce an exposed holdout; they are not
new independent evaluations.

## Final audit replay

The final source/output review reused the previously clean-installed Linux
environment. All 23 package pins matched and `pip check` passed. Tests passed
again: **186 in 81.64 s**, warnings as errors. All seven default CLI stages and
explicit CSV prediction completed. Dataset CSV hashes and final JSON summary
matched the original run exactly; all 29 frozen source/config/protocol files
were unchanged. Independent calculations passed 432 recorded comparisons.

The supplied post-documentation Windows screenshot also reports:
`Frozen executable files and evaluation protocol: UNCHANGED`.

The report builder ran separately on Linux/Python 3.12.14 with ReportLab 4.4.9,
Pillow 13.0.0 and charset-normalizer 3.5.2. Its optional dependencies are in
`report_tools/requirements.txt`; they are not added to application requirements.
Windows report generation has not been executed. See
[FINAL_AUDIT.md](../reports/FINAL_AUDIT.md) for current verification scope.

## Boundaries

Other Python versions/platforms are not claimed tested. The model loader checks
Python major/minor and exact NumPy, pandas, SciPy, scikit-learn and joblib versions.
These guards do not guarantee binary portability or authenticate pickle files.
Train on the target machine. A pinned snapshot is not a hash-locked build.

The evaluated source, config and FINAL_EVALUATION_PROTOCOL.md remain byte-identical.
Some frozen source docstrings retain stage-era wording; current scope is in
README and the report. Cosmetic source edits were avoided after the freeze.
Historical stage documents are labeled accordingly.

Real-shipment accuracy, production load, missing-data imputation, unseen-route
performance and calibrated individual ETA intervals are not verified.
