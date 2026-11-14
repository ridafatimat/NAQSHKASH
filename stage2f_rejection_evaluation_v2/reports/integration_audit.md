# Dataset-v2 rejection integration audit

The full source is exactly:
C:\NAQSHKASH\stage2\stage2f_negative_dataset_v2\dataset\full_20261106_194834_776.
Its manifest/completion describe4000 images/1000 parents; selected development
metadata describes2800 images/700 parents. No image bytes were opened during
the implementation's metadata compatibility audit. The prior post-seal report
at reports/20261106_194834_776_full_validation/validation.json records3400
semantic checks and600 protected hash-only checks. That prior result is not a
new evaluation executed here. This workspace never invokes that whole-dataset
validator, since it would touch validation/protected files.

## Frozen interfaces actually reused

- stage2f_negative_rejection/src/evaluate_baseline.py: FrozenPipeline and
  load_recognizer. `.observe(raw_bytes)` performs exactly the existing D1/E1
  operations and returns preprocessing status, D1 proposals/roles/boxes/path,
  confidence and per-tier greedy predictions. It takes no transcription target.
- stage2d_experiments/pipeline_d1.py: process_image -> prepare/process_prepared;
  frozen D1.json remains the existing config. No extra segmentation rule,
  threshold, geometry normalization or crop correction is introduced.
- E1 selected checkpoint:
  C:\NAQSHKASH\stage2\stage2e_finetuning\checkpoints\selected_model\best.pt.
  SHA256:27f2da50795e2fca8f327afefdce10645f3e60e5320060a8cee93994fabdcdf1.
  Original CRNN(66), fixed65-character vocabulary plus blank0, strict state-dict
  loading, eval/requires_grad(False), torch.inference_mode. Grayscale/255,
  bilinear aspect-preserving height64 and original greedy CTC are unchanged.
  Original single-tier forward avoids changing padding/context through batching.
- Existing confidence.py from_logits/page_signals is called INSIDE FrozenPipeline,
  without target annotations or calibration. Raw scores remain uncalibrated.
- stage2f_negative_rejection/src/rejection_pipeline.py and
  configs/f1_ocr_uncertainty_policy_v2.yaml: original F1 v2 unchanged. OCR-only
  wrong-tier/unsupported glyph findings produce REVIEW, not proven image invalidity.
- stage2f_tier_aware_decoding/src/contextual_pipeline.py: original process(),
  role_evidence(), tier_aware_decoder.interpret(), verified registry and rendering
  profile. R0 invokes disabled context; R1 invokes enabled context. Its existing
  unresolved-context guard is retained. No decoder or policy source is modified.
- T1 measurements.page_errors and assess_corrections: valid-reference edit and
  all-minimum-alignment correction accounting, reused in evaluation only.

Imports preserve the original T1 runtime/common bindings; the legacy pixel
module is loaded with a unique module alias to avoid collision with this new
evaluate_rejection.py. No second OCR architecture or copied model code exists.

## Data boundary and T1 provenance

The loader first gates development_design, before any image/annotation file
access. It checks full manifest ID/hash/parent ownership metadata, then validates
only selected development labels, pairs, semantic lineage, metadata and (when
requested) image bytes/hashes. There are no dataset image-directory globs.
Quarantine markers, unsupported categories, unverified statuses, fake invalid
CTC targets, pair mismatches, metadata drift and historical/preflight semantic
collisions fail closed. Reserved preflight/quarantine parent indexes and the
completed full historical semantic index are read as metadata only.

For each image, the original pixel API receives only bytes. Saved raw observations
contain original identifiers/lineage and a whitelist of rendering provenance,
not validity, expected outcomes, reference text or mutation-rule labels. T1 gets
only observation + rendering context. Dataset layout supplies font hash/source48
and the frozen renderer's grayscale integer-lt convention; source modeL is a
property of that locked renderer, not inferred from validity. Only known pointwise
illumination-PNG recipes inherit source provenance. Unknown font/size/convention
or recipe yields empty context. Ground-truth rows/geometry are never substituted
for D1's predicted tiers or boxes, and the dataset inverse verifier is NEVER used
as a runtime rejection oracle.

The existing F1 validator is conservative and checks observable decoded syntax.
It does not gain the dataset verifier's full-grid count/alignment proof. A bad
image can produce plausible valid OCR. R0/R1 must therefore be measured honestly;
no true-invalid-rejection improvement is implied by implementation or fewer reviews.

## One inference pass, replay and errors

The runner is manual. CPU is default; cuda/auto reuse the old loader. Per-record
seeds and exact device/library provenance are recorded for deterministic resume.
Each original image/tier is processed singly; no model/threshold/normalization
is changed. Immutable per-record files are committed atomically, with payload
and input lineage hashes and an OS run lock. Resume skips committed records.
An interrupted uncommitted record can be recomputed; there is no second full OCR
pass. Once coverage is complete a single immutable observations.jsonl and
completion marker are sealed. Both candidates require that exact raw SHA and
record set. No tensors or whole-dataset image copies are stored.

Known returned preprocessing/segmentation/OCR statuses are preserved. Uncaught
ValueError/RuntimeError/OSError during observe are recorded as unlocalized D1/E1
pipeline exceptions and REVIEW-compatible observations, not proof of invalid
content. They are reported separately rather than falsely localized as D1 or
OCR failures. Programming errors/model initialization problems abort the run.
Committed failures are not silently retried/cherry-picked by resume.

## Evidence limits

Labels are AUTO_VERIFIED under the source's fixed-font canonical/raster scope;
no Talim expert review occurred. Illumination-only variants are not a broad
photo/scan benchmark. Development siblings remain correlated; image rates and
parent macro/worst cases are separate. No validation/protected image access,
model inference, full R0/R1 evaluation, training or calibration was executed
during implementation. Only tiny fixtures/mocks and metadata/hash audits ran.
