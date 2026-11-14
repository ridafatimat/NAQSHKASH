# Stage 2F evaluation protocol (frozen before inference)

The negative source family is assigned entirely to evaluation-only diagnostic
coverage. It has no negative development counterpart: splitting mutations or
augmentations across partitions would leak the source parent. These evaluation
fixtures are historically exposed adversarial data, NOT an independent test.

Positive edge controls use the historical Stage2D tuning vs validation assignment:
tuning goes to development; validation goes to evaluation. E1 already used these
for training/selection, so both are exposed evidence. Clean synthetic historical
training regression parents split by deterministic SHA256 parity of parent ID.
All variants of a parent stay together. Frozen positive test assignments are
excluded entirely; no test images are read or copied. General scanned exploratory
parents are not included in this focused clean/positive control manifest.

Source annotations are OFFLINE scoring data only. They never enter preprocessing,
OCR, structural decisions or confidence. A source's fixture category never drives
runtime decisions. All 198 fixture records (197 files + one missing image) remain
in the inventory, including metadata-only cases. Metadata-only fixtures are scored
as valid image controls where pixels match valid base content. Unsupported/ambiguous
image representations expect REVIEW, and are outside binary content metrics.

F0: D1 proposal + E1 greedy OCR; unreadable/no-ink input rejected, failed D1
proposal sent to REVIEW (D1 itself only exposes proposal accepted/not accepted).
F1: F0 plus observable tier checks; stripped tier offsets prohibit cross-tier
character-position checks. Aligned canonical validation is available only when
true alignment is explicitly supplied. F2 adds configurable quality thresholds;
low confidence causes REVIEW, never a claim of proven invalid content.
F3 absent because no classifier experiment is justified.

Three outcomes are always retained. Rejection recall = REJECT_INVALID / invalid
image inputs; FAR = ACCEPT / invalid inputs; review rate reported separately.
Positive acceptance = ACCEPT / valid images; false rejection = REJECT_INVALID /
valid images; positive review = REVIEW / valid images. Nonacceptance = reject +
review, separately named. Unsupported/ambiguous/unadjudicated/missing entries
are excluded from binary metrics and remain in the three-way/expected-outcome
tables. An absent artifact is not an attempted OCR image. Metrics distinguish
artifact-invalid from Talim-content-invalid; absent content denominators are null.

Accepted-output correctness = exact nonempty tier text + row/role agreement with
source canonical annotation among accepted valid pages with scorable truth.
Leading blank cells are stripped independently exactly as E1 targets; internal
spaces remain significant. This verifies transcription/roles, not pixel alignment.
End-to-end page correctness requires ACCEPT and exact complete row/role/text
agreement for valid pages, or the expected action for invalid/uncertain inputs.
Wrong segmentation cannot be hidden by discarding failed pages. Counts are
page/image weighted plus macro parent rates; variant duplicates are not independent.
Reason accuracy is null unless image-level reason-code labels were independently
provided; source mutation labels are not runtime reason ground truth.

Thresholds may use development outputs only. The calibration tool checks the exact
frozen development manifest, image IDs, input hashes and lineage. Default content
calibration refuses this dataset because it has no independent invalid development
parents. An explicitly requested --allow-positive-only quality gate uses a positive
development quantile and is labeled quality-only, NOT calibrated validity or
negative recall optimization. The evaluation partition is never used to select it.
No classifier or recognizer training is performed by any Stage 2F script.
