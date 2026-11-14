# NAQSHKASH Stage 2F — negative edge-case rejection

Implemented preparation and manual experiment framework. **No training,
full inference/evaluation, or calibration was executed during implementation.**
E1 remains frozen and unchanged. See `RUN_IN_VSCODE.md` for executable commands.

The audited “negative” data are validator fixtures: 49 categories, 51 source
records, 50 clean image files and 147 scanned variants, **one independent source
family**. 46 clean images have valid rendered content but negative annotations;
45 are byte-identical to the base. One file is corrupt, one blank, one has
ambiguous resized dimensions, one uses RGB, and one source image is missing.
No image-invalid Talim-content collection was established. A learned classifier
is not justified and no classifier modules/checkpoints are produced.

Implemented modules:

- `prepare.py`: inventory, source/hash audit, frozen parent-disjoint manifests,
  leakage documentation, HTML gallery and scoped historical input locks.
- `evaluate_baseline.py`: label-blind D1/E1 pixel inference and separate offline
  scoring. Supports CPU/CUDA and explicitly selected smoke samples (maximum 3).
- `talim_validator.py`: original canonical annotation validator, authentic
  aligned-text checks, and conservative OCR tier syntax validation.
- `rejection_pipeline.py`: F0/F1/F2 three-way decisions.
- `confidence.py`: CTC forward sequence score for the greedy target, frame/path
  scores, top-two frame margins and D1 ambiguity signals. These are heuristics.
- `calibrate.py`: development-only gate preparation; reports current content
  calibration limitation, or explicit positive-only quantile quality gating.
- `evaluate_rejection.py`: saved-output decisions, metrics and stratification.
- `compare_candidates.py` and `report_generator.py`: paired saved-result comparison
  and Markdown reports; refuse incomplete/drifted/unpaired runs.
- `common.py`: bounded read-only historical integration and no-overwrite I/O.

All source annotations remain unchanged. Metadata labels do not enter image
decisions. REVIEW is never silently merged into REJECT. “ACCEPT” means passed
implemented observable checks, not proof of image validity. A failed model job
has no completion marker and cannot enter comparison/calibration.

Manifests use `archive_absolute_path::member` for archived images, avoiding
dataset extraction or source changes. Configuration `.yaml` files use the JSON
subset of YAML 1.2, so no PyYAML dependency is needed.

Audit/protocol outputs are actual asset/metadata findings. Empty metrics and
checkpoints directories are intentional until manually executed experiments.
Future content-rejection calibration needs independently authored, adjudicated
invalid images in multiple parent-disjoint development/evaluation families.
