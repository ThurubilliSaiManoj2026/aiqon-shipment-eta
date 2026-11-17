# Stage 5 verification record

> Historical stage record: statements about pending work describe that stage.
> For the completed implementation, current results and verified environments,
> see [README](../README.md), [ENVIRONMENT](ENVIRONMENT.md) and
> [REQUIREMENTS](REQUIREMENTS.md). Preserve pre-evaluation decisions as history.

## User environment evidence

The user supplied Windows output showing 48 passing stage-4 tests with warnings
treated as errors, successful generation, and the same structural audit as the
local stage-4 run. No stage-5 Windows run has yet been observed.

## Local execution

Verified in a fresh Linux environment with Python 3.12.14, NumPy 2.5.4,
pandas 3.1.1, SciPy 1.18.1, scikit-learn 1.9.1, Matplotlib 3.11.2,
joblib 1.6.0 and pytest 9.2.0. Principal package versions match the user's
installation, but Windows and Python 3.12.1 remain a separate execution target.

- `python -m pytest -q -W error --tb=short`: **79 passed**, no warnings.
- Preparation CLI with warnings treated as errors: succeeded.
- 237 development IDs, 60 reserved holdout IDs, three excluded IDs.
- Fold training sizes 114, 153 and 194; each validation window has 40 records.
- Three-feature and nine-feature matrices have full column rank including the
  intercept on development data; neither contains a constant feature.
- Exactly 120 validation shipments per baseline; zero holdout IDs in exported
  predictions; no learned model fitted.
- Baseline ETA MAE: 10.656094322251958 h (scheduled) and
  7.145461748435929 h (carry-forward).

An initial compatibility test caught pandas' deprecated `set_index` argument.
It was removed while retaining explicit duplicate-ID validation. Tests also
cover dispatch intervals below timestamp resolution and simultaneous checkpoints
at a split boundary. All final tests pass without suppressing warnings.

## Test coverage added

Hand-calculated feature values and periodicity; row-local feature computation;
future-field rejection; schedule/carry-forward formulas and early arrivals;
negative prediction clipping; nonfinite and misordered predictions; target-ID
alignment; hand-calculated metrics; label availability at every cutoff;
strict equality exclusion; stable splits under row permutations; infeasible
split settings; dataset hash mismatches; unchanged input data; deterministic
outputs; no holdout scoring; command-line success and repeat-run failure.

## Limits

The results are validation scores from a simulator, not real-world performance
or final holdout estimates. The two feature groups are prepared hypotheses;
their benefit is not established until learned models are compared on the saved
folds. Model explanations, missing-input robustness and final reporting remain
later stages.
