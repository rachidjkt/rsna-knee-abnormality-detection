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
Keep PROGRESS.md, TASKS.md, and RESUME.md current. No remote push or billable services.

The current user request explicitly authorizes private publication of this portfolio edition to rachidjkt after content review. That authorization supersedes the historical no-remote-push instruction for this repository only. Verify owner, private visibility, and remote head; never force-push.
