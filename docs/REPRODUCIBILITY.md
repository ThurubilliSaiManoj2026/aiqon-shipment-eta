# Reproducing a completed project safely

The original and fresh Windows installations passed 186 tests with warnings
as errors. The supplied complete workflow transcript reports three dataset CSV
hash matches and exact equality of original/reproduced final summaries.
This is same-protocol reproduction of an exposed holdout, not a new test.

The tested fresh environment is `artifacts/stage10_env`. To create it from
scratch, first ensure the destination does not exist. Run each command separately:

```powershell
.\.venv\Scripts\python.exe -m venv artifacts/stage10_env
.\artifacts\stage10_env\Scripts\python.exe -m pip install -r requirements.txt
.\artifacts\stage10_env\Scripts\python.exe -m pip check
.\artifacts\stage10_env\Scripts\python.exe -m pytest -q -W error
```

Preserve existing output directories. In the same terminal set:

```powershell
$reproPython = ".\artifacts\stage10_env\Scripts\python.exe"
$reproRoot = "artifacts/stage10_reproduction"
Test-Path -LiteralPath $reproRoot
```

Continue only if that last result is False. If already completed, inspect the
existing results; do not overwrite them. For a deliberately new reproduction,
choose a new output root and record the reason. Never change seeds or settings
to search for a favorable holdout score.

Run each command separately and stop on any error:

```powershell
& $reproPython -W error scripts/generate_data.py --output-dir "$reproRoot/data"
& $reproPython -W error scripts/prepare_development.py --data-dir "$reproRoot/data" --output-dir "$reproRoot/stage5"
& $reproPython -W error scripts/train_models.py --data-dir "$reproRoot/data" --stage5-dir "$reproRoot/stage5" --output-dir "$reproRoot/stage6"
& $reproPython -W error scripts/predict.py --demo --data-dir "$reproRoot/data" --model-dir "$reproRoot/stage6" --split-manifest "$reproRoot/stage5/split_manifest.json" --output-dir "$reproRoot/stage7_demo"
& $reproPython -W error scripts/review_development.py --data-dir "$reproRoot/data" --stage5-dir "$reproRoot/stage5" --model-dir "$reproRoot/stage6" --output-dir "$reproRoot/stage8"
& $reproPython -W error scripts/freeze_evaluation.py --data-dir "$reproRoot/data" --stage5-dir "$reproRoot/stage5" --model-dir "$reproRoot/stage6" --review-dir "$reproRoot/stage8" --output-dir "$reproRoot/stage9_freeze"
& $reproPython -W error scripts/evaluate_holdout.py --data-dir "$reproRoot/data" --stage5-dir "$reproRoot/stage5" --model-dir "$reproRoot/stage6" --review-dir "$reproRoot/stage8" --freeze-dir "$reproRoot/stage9_freeze" --output-dir "$reproRoot/stage9_final"
```

All CLI arguments were checked against the implemented parsers. Equivalent
argument lists were executed locally with all 54 original data/artifact files
unchanged. The user subsequently completed these commands on Windows.

Compare outputs, adjusting the reproduction root only if explicitly changed:

```powershell
@'
import hashlib
import json
from pathlib import Path

root = Path.cwd()
repro = root / "artifacts/stage10_reproduction"
for name in ("journeys.csv", "checkpoint_inputs.csv", "targets.csv"):
    original = (root / "data/generated" / name).read_bytes()
    recreated = (repro / "data" / name).read_bytes()
    same = hashlib.sha256(original).digest() == hashlib.sha256(recreated).digest()
    print(f"{name}: {'MATCH' if same else 'DIFFERENT'}")
old = json.loads((root / "artifacts/stage9_final/summary.json").read_text(encoding="utf-8"))
new = json.loads((repro / "stage9_final/summary.json").read_text(encoding="utf-8"))
print("Final summary matches exactly:", old == new)
'@ | & $reproPython -
```

Expected observations from the verified runs: three MATCH results and True.
Investigate differences rather than editing reference evidence. Provenance
hashes can differ because manifests record environment/source-stage metadata;
binary model equality across platforms is not required. Numerical tolerance
may be appropriate for legitimate cross-platform rounding, but must be stated.

For future published-repository verification, use a genuinely fresh checkout
and environment. The current Windows run reused the delivered source folder
with a new environment and new outputs; it was not a GitHub checkout check.
