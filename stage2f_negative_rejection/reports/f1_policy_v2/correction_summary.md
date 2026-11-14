# F1 v2 targeted policy correction

Modified: src/rejection_pipeline.py and src/evaluate_rejection.py only.
Added: configs/f1_ocr_uncertainty_policy_v2.yaml, tests/test_f1_policy_v2.py,
tests/fixtures/f1_v2_development_cases.json, RUN_F1_POLICY_V2.md, and this
reports/f1_policy_v2 directory (protected_before.json, lightweight_tests.txt,
verification.json, correction_summary.md).

Previously the final pipeline returned the syntax validator's REJECT_INVALID
verdict directly. The canonical syntax verdict remains intact, but OCR-only
violations now support REVIEW_UNCERTAIN. Existing confirmed unreadable/no-ink
input artifact findings still support REJECT_INVALID and take precedence.
Multiple findings retain their source, scope, row/tier, predicted text/glyphs,
original evidence and reason code. No new image invalidity rules were introduced.
F0 decisions/schema remain unchanged. F2 uses the same v2 decision mapping.

36 lightweight tests passed, including four saved development-case regressions,
F0 compatibility, multi-reason precedence, provenance preservation and immunity
to evaluation-only ground-truth fields. The CLI --help check passed. No full
F1 evaluation or OCR ran. No evaluation-partition outcomes were inspected;
existing result files were read only as opaque bytes for integrity hashes.

All 23 newly protected historical files matched their before hashes. The existing
Stage2F scoped historical integrity test also passed (282 files). Original F0/F1
outputs, frozen manifests, canonical validator/rules, D1 preprocessing and E1
checkpoint are unchanged. verification.json records tested code/config hashes.
Policy semantics are versioned before application to evaluation outcomes;
review manually rerun development results before applying to evaluation.

Run the exact PowerShell block in RUN_F1_POLICY_V2.md yourself. It reads the
existing 20261106_154728_812_dev_baseline and writes a new timestamped
20261106_154728_812_dev_F1_policy_v2_* directory containing results.json,
summary.json, groups.json, per_image.csv and completion.json. The old
20261106_154728_812_dev_F1 directory is untouched. Full aggregate counts remain
unmeasured until that manual run.
