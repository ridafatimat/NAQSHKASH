# Negative Dataset v2 rejection evaluation — implementation only

R0: existing D1 -> frozen E1 -> unchanged F1 v2.
R1: exactly the same saved observations -> unchanged optional T1 -> F1 v2.

The explicit full dataset is dataset-v2/dataset/full_20261106_194834_776.
Only its development_design partition (2800 images/700 parents) is accessible
to these commands. Validation/protected images are blocked, including byte/hash
reads by these evaluation tools. Manifest ownership metadata is checked globally
without opening excluded images/annotations. No directory-based image discovery.

The runner reuses FrozenPipeline.observe, model loader, preprocessing, confidence
and greedy CTC; no second recognizer exists. Raw immutable record checkpoints
allow resume. One sealed observations.jsonl is reused by R0/R1. Runtime APIs get
pixels and whitelisted rendering metadata, never validity/reference labels or
true layout geometry. Unknown rendering context cannot force T1 corrections.

Semantic validity and transcription correctness are separate. Invalid-image CER
is never fabricated. REVIEW remains distinct from REJECT. Outputs include
per-parent macro/worst cases, category/variant/length/row groups, actual F1 reasons,
T1 edit correctness/unknowns, detailed failure tables and visual galleries.
No automatic threshold tuning, policy change or winner selection is implemented.

See reports/integration_audit.md, reports/evaluation_protocol.md and
RUN_REJECTION_EVALUATION_IN_VSCODE.md. All full inference/evaluations are MANUAL.
Implementation tests use tiny synthetic contracts and mocked observations only.
The actual source dataset has no human Talim expert adjudication, and illumination
variants do not establish broad scan/handwriting validity. No improvement claim
is made before actual results exist.

Created scripts: dataset_loader.py, run_ocr_baseline.py, observation_store.py,
evaluate_rejection.py, evaluation_metrics.py, compare_rejection.py,
generate_error_gallery.py, verify_artifacts.py and integration.py. JSON-subset
YAML configs require no new configuration dependency. Historical inputs/code and
new experiment files are hash-locked; output folders are new and never overwrite
the source datasets, original manifests, models or prior results.
