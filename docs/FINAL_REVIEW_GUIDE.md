# Final review: Windows and Cursor

The reviewed package is `aiqon_submission_review.zip`. It contains the full
source and final documentation, including the monochrome three-page PDF.
It deliberately omits virtual environments and generated model/data folders.

## Preserve the completed project

1. In File Explorer, right-click the downloaded ZIP and select **Extract All**.
   Extract into a new folder such as `Downloads/aiqon_submission_review`.
2. Open the extracted `aiqon-shipment-eta` folder. Its top level contains
   `README.md`, `src`, `scripts`, `tests`, `config`, `docs`, `reports` and
   `report_tools`.
3. Open `reports/Shipment_ETA_Technical_Report.pdf` and read all three pages.
   Open `reports/FINAL_AUDIT.md` in Cursor for the verification record.
4. To update the existing completed project, copy only `.gitattributes`,
   `README.md`, `docs`, `reports` and `report_tools` from this extracted folder into
   `C:\Users\sai manoj\Projects\aiqon-shipment-eta`.
5. Merge the folders and replace included documentation files when prompted.
   Keep the existing `.venv`, `requirements.txt`, `src`, `scripts`, `tests`,
   `config`, `data` and `artifacts` folders/files. There is no application-code
   change in this audit.

The full extracted package is also a clean source candidate for publication.
The existing working project retains the original training and holdout evidence.
Do not rerun workflow commands into completed output directories.

## Verify the documentation copy

In Cursor, open the existing project and its PowerShell terminal. Run:

```powershell
Set-Location -LiteralPath "C:\Users\sai manoj\Projects\aiqon-shipment-eta"
```

Then use the read-only freeze check in [STEP11_GUIDE.md](STEP11_GUIDE.md).
Its expected successful message is:

```text
Frozen executable files and evaluation protocol: UNCHANGED
```

That check has already passed in the supplied screenshot. Repeating it after
this copy verifies that the same frozen files were preserved. Stop and inspect
any mismatch rather than replacing the saved freeze. A documentation-only copy
does not require retraining or another holdout selection.

## Next submission step

Review [SUBMISSION_CHECKLIST.md](SUBMISSION_CHECKLIST.md), then inspect the actual
Git file list. The required remaining work is repository publication, recruiter
access verification and a new checkout check. Do not include environments,
private recruitment material or ZIPs in the repository. No publication or
submission has occurred as part of this audit.
