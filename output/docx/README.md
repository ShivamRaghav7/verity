# Complete EditGuard project report

`EditGuard_Project_Report.docx` is the complete technical report for the final
ReasonKE application, dated 29 October 2026. It contains 23 pages in the checked
Microsoft Word layout, six figures and fifteen tables. Pagination may change
with another editor or font environment.

The report covers the problem, method attribution, implemented architecture,
transactional knowledge lifecycle, controlled datasets, matched evaluation,
model configuration, measured results, failure analysis, application workflow,
deployment, reproducibility, references and a metric glossary. It keeps the
original operations study, separate depth challenge and website acceptance
workload distinct. It explains the observed perfect scores without turning
them into general accuracy guarantees or inventing local model training.

## Build from saved evidence

Use an isolated Python environment with `python-docx` and Pillow. The builder
audits the released receipts before writing the report and makes zero model
calls. The architecture image currently uses Windows Calibri fonts.

```powershell
python -m scripts.build_project_report_docx
```

The scientific figures are reused from the audited study export directories;
they are not redrawn with changed values. `report-build-manifest.json` records
their identities and the result and protocol source hashes.

The authoring script inserts a native Word table of contents. Its fields must
be refreshed in Word after a rebuild. The published copy already contains the
refreshed contents. The build command does not perform a layout review or
automatically reproduce the publication hash.

## Publication verification

Both studies were replay-audited and the India application receipt was checked
without inference. The final Word pages were rendered through the document
page enhanced-metafile API and inspected visually. The canonical LibreOffice
renderer was unavailable on this Windows host; Word PDF export stalled, so it
was not used as layout evidence. Only the final DOCX is the requested report
deliverable. Intermediate page images remain outside the published folder.

The publication manifest binds the delivered DOCX hash to its reviewed page
images. It records rendering limitations rather than claiming a PDF export
that did not complete. No frozen experiment, label, scoring rule or raw model
response was modified to produce this report.
