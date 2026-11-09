# NAQSHKASH Stage 2F — FYP report template

## 1. Negative Edge Cases objective

Detect or route invalid/uncertain Talim inputs safely after OCR, preserving a three-way ACCEPT / REVIEW_UNCERTAIN / REJECT_INVALID distinction. Human review is a deliberate outcome, not an autonomous invalid-image rejection claim.

## 2. Negative Dataset v2 design

4,000 synthetic fixed-grid images from 1,000 independent parent families, including matched valid/invalid controls and clean/illumination variants. Mutations change visible notation and were admitted by existing automatic raster/canonical verification. Domain-expert verification has not yet been established.

## 3. Supported invalidity categories

Wrong-tier placement, invalid count-token ordering, and missing required count. Only these three semantic mutation categories are covered; do not generalize to all malformed or handwritten Talim.

## 4. Split and parent independence

Development: 2,800 images / 700 parents. Validation: 600 / 150. Protected test: 600 / 150. Illumination variants are repeated observations of a parent, not independent sources. Describe frozen overlap/split checks and seals [insert artifact hashes].

## 5. Frozen D1/E1/T1/F1/R2 baseline

D1 preprocessing/segmentation, selected E1 CRNN-CTC checkpoint, optional scoped T1 alias interpretation and conservative F1/R2. R2 development: valid 1,376 ACCEPT / 24 REVIEW / 0 REJECT; invalid 788 ACCEPT / 612 REVIEW / 0 REJECT. No new OCR training is part of R3.

## 6. Root-cause investigation

Some image-level invalidity survives as valid-looking OCR; some count order and absence requires horizontal structure unavailable in stripped strings. Among 70 valid Stage 1 contradictions: 50 spacing-only, 12 count-character errors and eight symbol-character errors; 58 newly reviewed valid cases comprise 50 spacing and eight count errors. Do not weaken canonical rules to explain OCR mistakes.

## 7. Text-space method

Reuse bounded prefix-offset enumeration under the documented literal leading-stripped model, unchanged canonical checks and candidate-derived cross-row totals. Conditional search completeness is not image-geometry completeness. Historical Stage 1 diagnostic: valid 1,318 ACCEPT / 82 REVIEW / 0 REJECT; invalid 48 ACCEPT / 1,352 REVIEW / 0 REJECT. These were development-only REVIEW counterfactuals.

## 8. Image-component feasibility

The 48-image / 12-parent pilot matched 882/906 glyphs, with 0.5px median absolute position error. Associations were 502/502 correct among scored assignments, with only 502/534 visible counts scorable. All 8 expected empty tokens were detected and no false empty was observed, but eight valid controls had incomplete grouping. Wrong-tier identification was not established. The cache does not cover the 70 valid contradictions. These findings are promising, limited and not sufficient for definitive image rejection; components do not enter R3_HITL_V3 decisions.

## 9. R3 Human-in-the-Loop policy

R2 REVIEW/REJECT never becomes ACCEPT. R2 ACCEPT is retained with at least one adequately supported complete canonical page interpretation; uniqueness is not required. Contradiction/incomplete/unsupported evidence requires REVIEW. No new text-only REJECT. Report actual development differences rather than forcing reference counts.

## 10. Explainable findings

Findings retain OCR/candidate evidence and original R2 reasons. Internal row/column indices are zero-based, human descriptions one-based. Tokens, associated symbols and positions are populated only to supported scope. Total conflicts do not identify a guilty row. No annotation or mutation feature enters the policy.

## 11. Reviewer overlays

Optional small gallery uses verified cached D1 transforms and source identity, with broad approximate row/tier highlights. No text-column-to-pixel conversion, no generator-derived rectangles, and no guessed exact glyph location. Missing geometry produces text only. Decisions are sealed before image bytes are read.

## 12. Development evaluation

R3 actual confusion matrix/rates, transitions, new/preserved reviews, zero new rejects, category/parent/variant/transcription results and every Stage 1 difference: [PENDING MANUAL EXECUTION]. Describe the valid-review burden and residual unsafe acceptance honestly.

## 13. Localization evaluation

Report eligibility, conditional/full-denominator row/tier/candidate-column metrics, broad-region overlap and unknown/incorrect cases [PENDING]. Precise pixel mutation localization has zero coverage. Valid review regions have availability, not invalid mutation accuracy.

## 14. Validation and protected test

Approved freeze manifest/adapter hashes: [PENDING]. Validation results: [NOT EXECUTED]. Protected-test results: [NOT EXECUTED]. Use the separate authorized once-per-split protocol. No tuning after exposure; report any decline.

## 15. Limitations and future improvements

REVIEW is not definitive REJECT. Valid images may be reviewed because of OCR spacing/character errors. Wrong-tier substitutions may still produce plausible text. Only three invalidity categories and synthetic fixed-grid imagery are supported by this evidence. Domain-expert verification remains unestablished. CTC timing is unsuitable for direct glyph localization, including leading labels emitted before their ink. Component positioning has limited coverage and does not establish glyph identity or reliable source absence. R3 does not achieve autonomous invalid-image rejection.

## 16. Transition to real handwritten Talim fine-tuning

Plan separately curated/expert-adjudicated handwriting data, independent splits, OCR/segmentation adaptation and a new evaluation protocol. This milestone performs no handwriting training or model modification. Preserve frozen Stage 2F results as a synthetic-development baseline, not a handwriting accuracy claim.
