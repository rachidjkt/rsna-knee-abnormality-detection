# Project guidance

Preserve official metadata in data/. Never download the full MRI archive locally.
Use synthetic DICOM fixtures for local integration tests; mount competition inputs on Kaggle.
Do not print or publish radiology reports or private identifiers in documentation.
Unknown target labels are NaN, never negatives. Train the image-only baseline on explicitly labelled studies.
Use disjoint study splits; group by reliable patient identifiers if introduced. Reports must not become validation ground truth.
Keep synthetic metrics clearly separate from medical and competition performance.
Do not download model weights during submission inference. Record external weight provenance and eligibility.
Run pytest and the synthetic smoke workflow after material pipeline changes.
Regenerate notebooks with scripts/build_notebooks.py after source changes; notebooks embed the source for offline execution.
Keep PROGRESS.md, TASKS.md, and RESUME.md current. No billable services. Remote publication requires explicit user authorization.

The November 16 request authorized the initial private publication. The November 20 request authorizes documentation and metadata updates to this now-public repository. Verify owner rachidjkt and the current remote head; preserve public visibility and Git history, and never force-push. Do not change application code, evaluation settings, or saved scores for documentation work.
