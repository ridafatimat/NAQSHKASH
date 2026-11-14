# Image-component feasibility
Date: 2026-11-07. Implementation/specification stage; real cached-crop diagnostic remains **MANUAL, NOT EXECUTED**.

## Minimal independent observation

src/spatial_diagnostic.py components reads only the existing processed crop PNGs, D1 tier roles and crop boxes. It does not invoke D1, E1, T1 or a Talim policy. The pixel-only API receives no transcript, labels, renderer layout, font size, mutation or matched-control identity.

The diagnostic uses per-crop Otsu foreground extraction (standard deterministic image processing; no threshold sweep or label-based tuning), keeps every connected component without an area cutoff, and groups ink by contiguous occupied x-projection intervals. Multiple disconnected dots/bars whose x projections overlap are kept in one ink group. Separated fragments are not silently fused using source cell width. Component counts are not glyph counts.

Observed symbol-group left edges define candidate horizontal corridors. Count-group centers are assigned to the preceding observed symbol anchor. Equal anchors are ambiguous; unassigned ink is recorded. Empty corridors are candidates for investigation, never proof that a count is absent.

## Offline evaluation, after observation

Cached source-renderer reference boxes enter only scoring:
- one-to-one image-group/reference-glyph matches, splits, merges and uncovered glyphs;
- source-x pixel and width-normalized localization errors;
- conditional symbol/count association accuracy **and full visible-count denominator**;
- expected empty tokens, empty-corridor detections, false empties on valid controls and unscorable empty candidates;
- validity/category/parent/clean-illumination results and incomplete-grouping control counts.

All 906 source reference glyphs remain in coverage denominators; no gain may be claimed by dropping fragmented glyphs. Labels and generator coordinates cannot select a runtime grouping. Known source-frame transforms map evaluation boxes; no model bias is subtracted.

The algorithm produces **no character identities**. Wrong-tier identification accuracy is explicitly null/unestablished. It may locate ink that merits an independent identity check, but cannot distinguish D from _ semantically or reject it. An eventual glyph verifier must handle the audited aliases and unknown renderings separately.

## Verification run here

Ten lightweight synthetic tests passed: overlapping multi-component glyphs, separated-dot fragmentation, merged glyphs, constant crops, label-blind observations, correct/incorrect occupancy corridors, equal-anchor ambiguity, exact CNN mapping math and latest-completed-run selection. A synthetic valid control with lost crop ink produces a false-empty candidate; this demonstrates why empty corridors cannot alone trigger REJECT.

No real cached crop was processed by the component subcommand during this task. Accordingly **no measured component-method accuracy, missing-count recall or valid-control improvement is claimed yet**. Run the exact command in r3_implementation_spec.md to obtain metrics.json, per_image.json and report.md from the 48 cached cases, with no inference.

## Coverage limitation

The cache contains 24 valid and 24 invalid pilot images, but **zero of the 70 valid Stage 1 contradictions**. It overlaps eight of the 32 residual wrong-tier images and two of the eight residual missing-count images, and none of the eight residual count-order images. Even a favorable pilot cannot establish recovery of the 58 new valid reviews or efficacy on all 48 residual accepts.

After the manual cached test, source-crop evidence for the uncached target cohorts is the next data requirement. A separate D1-only extraction from existing development images could provide it without OCR, but it requires its own scope and was not performed. Do not generalize short/medium pilot safety to uncached cases.
