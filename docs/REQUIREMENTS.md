# Assignment traceability

Source: the complete three-page Aiqon assignment and supplied recruitment
emails. The user reported HR-confirmed deadline flexibility.

| Requirement | Evidence | Status |
| --- | --- | --- |
| Final shipment ETA | Checkpoint pipeline, CLI and timestamp tests | Implemented and verified |
| Delay likelihood OR magnitude | Nonnegative magnitude; signed deviation retained | Magnitude implemented and verified |
| Explain contributors | Exact terms/groups, reference, explicit delay conversion, report example | Implemented; noncausal limits stated |
| Relationships between stages | Arrival deviation, congestion interaction, timetable and recovery/missed-slot simulator | Implemented; measured comparison |
| Public OR synthetic data, around 100-300 | 300 fixed synthetic shipments on five routes | Validated and reproduced |
| Define ETA, delay, synthetic truth | DATA_SPECIFICATION.md, simulation TOML, report | Documented |
| Define evaluation | Label-available chronological splits, baselines, frozen holdout and uncertainty | Completed locally and on Windows |
| Clear repository instructions | README and REPRODUCIBILITY.md | Commands verified; published repository review pending |
| GitHub repository link | Accessible final repository | Pending publication/access check |
| Technical PDF, 2-3 pages | reports/Shipment_ETA_Technical_Report.pdf | Three pages prepared; candidate review pending |
| Optional demo, <=5 minutes | CLI demonstration available | Video optional; not recorded |
| Submit repository link and PDF to info@aiqon.in | Submission checklist | Pending; nothing sent |

Suggested challenges are not all mandatory. Route-related attributes and
congestion interactions are modeled; incomplete/invalid inputs are rejected.
Imputation and unseen-route generalization are not claimed. No UI, API, cloud,
LLM, probability classifier or multiple checkpoints are mandated.

The company emphasizes understanding, feature design, suitability, explanations,
logistics behavior and limitations. No accuracy threshold or ranking formula
was supplied. Test counts and synthetic results cannot guarantee a top-5% rank.
