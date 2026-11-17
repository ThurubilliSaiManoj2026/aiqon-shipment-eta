# Step 4 - run the synthetic-data generator on Windows

> Historical stage record: statements about pending work describe that stage.
> For the completed implementation, current results and verified environments,
> see [README](../README.md), [ENVIRONMENT](ENVIRONMENT.md) and
> [REQUIREMENTS](REQUIREMENTS.md). Preserve pre-evaluation decisions as history.

This update assumes the previous scaffold is already in the project and its
three environment tests passed. No additional package installation is required.

## Install the update

1. Download `aiqon_step4_update.zip`.
2. Right-click it in File Explorer and select Extract All.
3. Set the destination to your existing `aiqon-shipment-eta` project folder.
4. Allow replacement of exactly these updated documentation files:
   `README.md`, `docs/DATA_SPECIFICATION.md`, `docs/REQUIREMENTS.md`, and
   `docs/ENVIRONMENT.md`. Other delivered files are new to the step-3 scaffold.

The update does not contain `.venv`, `requirements.txt`, or generated data.
Keep the existing simulation TOML and environment test/configuration files.

## Run tests

In Cursor's PowerShell terminal, from the project root:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -W error
```

The updated suite contains 48 tests. If any fail, retain the full error output
and resolve it before generating the base dataset.

## Generate data

After the suite passes:

```powershell
.\.venv\Scripts\python.exe scripts/generate_data.py
```

The command prints `Validated dataset written to:` followed by an audit. Its
output folder is `data/generated` under the project. It creates:

- `journeys.csv`: full journey for auditing and training-label preparation.
- `checkpoint_inputs.csv`: only allowed prediction-time fields.
- `targets.csv`: IDs, label availability and target quantities.
- `audit.json`: structural counts, distributions and split feasibility.
- `simulation_config.json`: exact resolved configuration.
- `manifest.json`: seed, versions and hashes for provenance.

Do not overwrite generated data manually. A second run to the same destination
is intentionally refused; choose a new destination for an intentional repeat:

```powershell
.\.venv\Scripts\python.exe scripts/generate_data.py --output-dir data/reproduced
```

Share the test result and complete printed audit for review. Model training is
the next phase after the dataset is accepted, not part of this command.

## Interview explanation

"I generate complete journeys, then explicitly select only inputs available at
port arrival. The simulator includes upstream lateness, variable port processing,
dispatch-slot waiting and delivery buffers. I validate the chronology, labels,
CSV round trips and reproducibility. These synthetic results demonstrate the
method under documented assumptions; they do not establish real-world accuracy."
