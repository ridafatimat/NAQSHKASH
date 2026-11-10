# Final preflight validation

The FINAL collection is dataset/preflight, with120 model-input images from30
independent canonical source families:30 clean invalids (10 per category),30
matched valid controls and60 paired illumination-style variants. The independent
verifier established60 INVALID and60 VALID images;0 NEEDS_DOMAIN_REVIEW flags,
0 duplicate images,0 source collisions,0 split violations. Human-reviewed:0.
All images still await the user's visual/domain-scope review before full generation.

AUTO_VERIFIED is proof under the audited fixed-font canonical implementation,
not expert adjudication of all Talim dialects. All parents are development_design.
No protected test image, full4000-image dataset, full historical sweep, model
inference, training or threshold calibration was executed.

Visual QA caught a weakness in the INITIAL120-image provisional collection:
exact raster equality did not exclude all perceptual opposite-tier lookalikes.
That collection is preserved in dataset/quarantined_preflight_v0 with a
QUARANTINED marker; validation refuses it. It is NOT eligible for training or
rejection evaluation. The final gate excludes8 dot/bar/hook alias families and
independently considers their possible valid labels. Final wrong-tier parents
use only the14 visibly distinct supported shapes. See glyph_observability_audit.md.
Do not merge the two collections or count repeated parent sources as new parents.

No unchanged/alternate-valid sample entered the final invalid collection.
Tests explicitly exclude unchanged JSON-only defects, exact B/X alternatives,
near-identical C/Y alternatives and valid count changes. Actual final source
candidates excluded for semantic collision:0. These unit-test probes are not
failed dataset images and must not inflate exclusion counts.

Checks include original check_sample, production-versus-isolated pixel identity
for controls, all-valid-interpretations inverse parsing, pixel differences,
raw annotation/raster agreement, lossless illumination transfer, parent ownership,
duplicate and integrity checks. See the preflight construction_validation.json.

Open galleries/preflight.html. Full-resolution mutation zooms and original-image
links keep dense/long examples readable. Captions are outside model PNGs. Review
manual_review_checklist.md; no human/expert approval has been claimed.
