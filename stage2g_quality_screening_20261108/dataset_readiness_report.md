# Stage 2G automated screening and readiness

Existing audit reused; no discovery repeated. Original audit and E1/Stage 2F remain unchanged. No OCR, learned models, training, fine-tuning, Git or protected-test model evaluation.

## Results

467 pass automated quality checks; 453 flagged; 0 objectively excluded; 920 require human verification. Flag counts overlap.
```json
{
  "total_images": 920,
  "quality_results": {
    "PASS_AUTOMATED_CHECKS": 467,
    "FLAGGED": 453
  },
  "human_verification_images": 920,
  "accepted_images": 0,
  "source_label_reviews": 30,
  "source_label_corrections": "NOT_ESTABLISHED; exact TXT/JSON consistency passed prior audit, physical source-label correctness unverified",
  "source_split": {
    "train": 24,
    "test": 3,
    "validation": 3
  },
  "automatic_flags": {
    "GEOMETRIC_RECIPE_RISK": 330,
    "LIKELY_CLIPPING_PROXY": 103,
    "SEVERE_BLUR_PROXY": 115,
    "PARTIAL_CROP_RECIPE_RISK": 30,
    "VERY_LOW_CONTRAST_PROXY": 7,
    "PARENT_REGISTRATION_UNRESOLVED": 27,
    "SATURATION_OR_BACKGROUND_PROXY": 2,
    "MISSING_OR_UNREADABLE_FOREGROUND_PROXY": 11
  }
}
```

PASS_AUTOMATED_CHECKS is a quality-screen result, never ACCEPT. Only decode/structure failure or spatially constant images (each color channel range <=1) are objectively excluded. Blur, contrast, clipping, registration and unreadability proxies are heuristic NEEDS_REVIEW flags. No verified transcription errors or required original-label corrections have been established. All 30 source-to-label mappings still need human verification; the prior exact TXT/JSON agreement is not proof that handwriting matches those labels.

## Methods and thresholds
Pillow structure verification and full decode, OpenCV decode. Metrics computed at <=900px longest side. Blur flag: Laplacian variance <20, or <80 with <0.15 of source reference variance. Contrast flag: grayscale p95-p5 <20. Extreme-pixel flag: >70% pixels <=2 or >=253 (can be legitimate white paper). These are documented conservative triage thresholds, not calibrated acceptance criteria.

ORB features, ratio test 0.72, RANSAC 3px homography; require >=16 inliers, >=45% inlier fraction, >=12% reference spatial feature coverage, convex mapped corners and mapped area ratio 0.15–6. Otherwise parent registration is unresolved, not a content error. Foreground proxy is adaptive Gaussian threshold (31px/C12), including notebook ruling/shadow. >1% reference foreground mapped outside image flags likely clipping; >20% mapped foreground unmatched within a 7x7 dilation flags likely missing/unreadable foreground. These cannot identify Talim symbols, recover clipped material, distinguish handwriting from ruling or prove exact parent identity. Wrong-parent/repeated-feature registration remains possible. Partial-crop and geometry recipe names add risk flags only. A comparison against an already cropped reference cannot detect content lost before that reference was captured.

20 archive original_photo PNGs are reused as photographed references. For the other 10 sources, existing source_image_inventory.json identifies remaining.zip JPG references; extracted copies are stored separately. Reference mapping is explicit and provisional, not an authenticated parent chain. Synthetic reference PNGs are copied only where present in the existing inventory; unavailable references are labelled unavailable. No raster rendering has been invented. TXT/JSON remain untouched.

## Review and split isolation
Priorities: 0 source-label/provenance check; 1 likely clipping/missing foreground/partial crop; 2 other quality/registration/geometry flags; 3 quality-pass images. Queue has 30 source review tasks plus per-image tasks for non-excluded images. All image decisions require a human; no automatic acceptance. Source approval can be reused to avoid repeatedly comparing the same label for every augmentation, but each derivative still needs content-preservation checking. Source-label correction requires observed, documented discrepancy, not an automatic proxy.

The existing 24/3/3 source split is preserved verbatim. Main dashboard includes train/validation only. Test images are in a separate isolated_test_review.html, not linked or embedded in the development dashboard. Queue marks ISOLATED_TEST rows and places them after development rows; aggregate quality CSVs contain split markers. A separate custodian may perform test annotation/provenance QA using the isolated page. Do not use held-out images/metrics to develop models or tune thresholds. No writer-independent evaluation is claimed.

Input hashes were checked unchanged after screening; input_integrity.json records them. Review all flags in automated_quality_report.csv / flagged_images.csv; original metrics and evidence are preserved. No images are deleted. The dataset is screening-complete, human-validation-pending, NOT ready for automatic E1 fine-tuning. Run test_screening.py for lightweight algorithm and output checks.
