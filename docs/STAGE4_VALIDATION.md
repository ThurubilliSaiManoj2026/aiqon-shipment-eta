# Stage 4 validation record

> Historical stage record: statements about pending work describe that stage.
> For the completed implementation, current results and verified environments,
> see [README](../README.md), [ENVIRONMENT](ENVIRONMENT.md) and
> [REQUIREMENTS](REQUIREMENTS.md). Preserve pre-evaluation decisions as history.

## Implemented

- Strict TOML configuration validation, including unknown-key rejection.
- The scheduled and actual journey equations from DATA_SPECIFICATION.md.
- Nanosecond dispatch-slot arithmetic, including exactly-at-slot readiness.
- Balanced routes and deterministic, isolated random generation.
- Complete journey validation and a strict checkpoint-input allowlist.
- Three data tables, structural audit, configuration snapshot and provenance.
- CSV round-trip checks and refusal to overwrite an existing output directory.
- Unit, integration and end-to-end command-line tests.

## Executed checks

The user's Windows stage-3 setup passed its three environment tests, as shown
in the supplied terminal screenshot. Stage-4 tests have not yet been executed
on that Windows computer.

Local verification used a fresh Linux virtual environment with Python 3.12.14
and these matching principal package versions from the user's installation:

| Package | Version |
| --- | --- |
| NumPy | 2.5.4 |
| pandas | 3.1.1 |
| SciPy | 1.18.1 |
| scikit-learn | 1.9.1 |
| Matplotlib | 3.11.2 |
| joblib | 1.6.0 |
| pytest | 9.2.0 |

- `python -m pip check`: no broken requirements found.
- `python -m pytest -q -W error`: **48 passed**, with no warnings.
- Default generator command: succeeded and validated CSV round trips.
- Same-config/seed repeated exports: matching CSV and audit bytes, checked in tests.
- Tests include three additional seeds; these are invariant checks, not model
  performance selection experiments.

The matching-version check initially found a deprecated pandas CSV parser
option. It was removed; the final warning-as-error test run passed.

## Observed default-seed data audit

These are synthetic-data properties, not prediction results.

| Check | Observed result |
| --- | ---: |
| Shipments / unique IDs | 300 / 300 |
| Routes / shipments per route | 5 / 60 |
| Missing values | 0 |
| Early / exactly on-time / late final arrivals | 89 / 0 / 211 |
| Late port arrival but final delivery not late | 53 |
| Late port arrival with greater final lateness | 110 |
| Mean final delay magnitude | 9.156341 hours |
| Maximum final delay magnitude | 62.247973 hours |
| Median remaining duration at checkpoint | 39.696722 hours |
| Proposed holdout checkpoints | 60 |
| Earlier shipments with labels already available | 237 |
| Earlier shipments excluded for unavailable labels | 3 |

Zero exactly-on-time shipments is consistent with continuous random durations
and the strictly-zero definition; no operational grace period was specified.
The simulator parameters were not tuned after seeing these counts.

## Boundaries of these results

- Linux is not Windows; the user's Python patch version is 3.12.1, not 3.12.14.
  Run the supplied suite on Windows before accepting that environment's result.
- Matching seeds are reproducible within a fixed environment and implementation;
  exact cross-platform CSV hashes are not promised.
- No model has been trained or scored. These tests establish implemented
  mechanics and invariants, not real-world validity or predictive accuracy.
- Optional missing-input experiments, route-generalization policy, feature
  engineering, model selection, explanations and final reporting remain later work.
- The structural split audit is not yet a persisted model split/fold manifest.
