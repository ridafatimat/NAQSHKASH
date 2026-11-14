# Negative-fixture root cause analysis

This is diagnosis of an already exposed evaluation, not independent test evidence.
Read-only inputs: C:\NAQSHKASH\stage2\stage2f_negative_rejection\metrics\20261106_154728_812_eval_baseline and C:\NAQSHKASH\stage2\stage2f_negative_rejection\metrics\20261106_final_eval_F1_v2. All 434 saved rows were joined by ID;
180 wrong-tier reviews and 198 negative-fixture records were audited. No model,
preprocessing, segmentation search, rejection policy or calibration was executed.

## Confirmed findings

The excessive review rate is driven by one repeated source family, not 178
independent failures. Of 180 wrong-tier reviews, 178 are negative fixtures (47
clean + 131 scanned), and 2 are ordinary clean positives. There are 3 independent
parent identities total, 135 distinct file hashes among these 180 decisions,
and 46 affected negative mutation categories. The 178 fixture cases still have
only one independent source family, with two derived canonical content templates.

| Diagnosed mechanism | Reviewed images | Independent parent families |
|---|---:|---:|
| Expected singleton B (UP) decoded X (DOWN), identical visible font mask | 176 | 2 |
| Of the above: negative-fixture copies/variants | 175 | 1 |
| 2x2 black-patch variants interpreted as count-only X/Z | 3 | 1 (same fixture family) |
| Ordinary positive lower-tier [ and _ decoded G and L | 1 | 1 |

The family counts overlap and must not be added across rows. The final row's
parent is different from the singleton-B ordinary-positive parent.

## The B/X result is not evidence of an image-level wrong-tier symbol

The supplied Talamat.ttf encodes B and X with byte-identical 10x8 visible masks
at font size 48. Their full font masks have different blank extents; D1 crops
ink bands with small margins, and the recognizer resizes each crop to height 64.
The E1 call receives only the crop tensor; row/tier role is recorded externally
but is not supplied to the CRNN. Thus a single dot cannot intrinsically identify
B versus X from its shape. Correct interpretation needs the upper/lower context.

The base image visibly contains A11/R19 and S20/B10. The second row upper is
the singleton dot representing B; saved D1 bands put it at y194..202 and its
upper crop at [20,190,128,206] (height16, resized width432). The saved row/role
layout agrees with the canonical two-row layout in all 175 B/X fixture cases.
The same ordinary positive failure has upper B at row2, band y364..372, crop
[22,360,372,376], correctly labeled upper. No tier swap is demonstrated.

There is one singleton-B upper tier among the 236 ordinary positive controls;
it too decodes X. For B-containing multi-symbol upper tiers, 29 are transcribed
exactly, 3 are not exact, and one belongs to a failed segmentation proposal.
These are tier counts, not independent image accuracy estimates. This contrast
supports missing role/context plus short glyph-only input as the concentration
mechanism. Repeating that base hundreds of times amplifies this ambiguity.
No inference experiment was run to quantify a fix or isolate every internal
model factor; causal claims beyond the demonstrable information loss remain
untested. We do not claim all ordinary positive OCR errors are explained by B/X.

The other ordinary failure has correctly assigned upper/count/lower rows but
lower [ and _ predicted G and L. Original pixels and canonical annotation agree;
the glyphs are visually similar but their tight font masks are not identical.
This is an OCR transcription substitution, not an established segmentation swap.

## Dataset construction and image/annotation disagreement

The recovered original generator is C:\Users\hp\OneDrive\Documents\ChatGPT\FYP\edge_audit_staging\dataset_tools\edge_case_audit.py, negatives(), lines227..284.
The deployed C:\FYP Dataset Generation\dataset_tools directory no longer has
edge_case_audit.py; the retained project staging copy is identified explicitly,
not represented as a deployed active generator. Its mutation operations agree
with the observed source artifacts and negative_expectations.json. Inputs and
generator hashes are recorded in protected_before.json.

The generator first writes a valid rendered base image and TXT, then most
mutations change JSON only. Wrong-line/count/symbol JSON mutations do not rerender
the PNG. TXT mutations replace text files only. Duplicate-ID and semantic-
duplicate cases add extra image records. The scan script explicitly augments
readable images regardless of malformed JSON, preserving this distinction.

Consequently fixture labels such as count_zero, symbol_unsupported and wrong_line
describe validator defects, not visible invalid notation. This is intentional
validator-test design, but is inappropriate binary image-rejection supervision.
Source JSON is not used by runtime OCR, so annotation mutations do NOT directly
cause the B/X predictions. They affect dataset interpretation/training eligibility;
repeated glyph-only crops explain the observed prediction concentration.

There are 49 mutation categories, 51 clean source records, 50 actual clean files
and 147 scanned variants: 198 records but 197 image files. 45 clean PNGs are
byte-identical to the valid base. RGB mode contributes one more identical
grayscale raster (46 clean rasters total). Two derived content templates remain
one family, not separate independent negative parents.

## Corrected diagnostic classification (historical manifests remain untouched)

| Image class | Clean files | Scanned files | Total files |
|---|---:|---:|---:|
| Valid content with annotation/validator-only defects | 46 | 138 | 184 |
| Valid content, source representation/mode fixture | 1 | 3 | 4 |
| Blank white input | 1 | 3 | 4 |
| Corrupt/unreadable PNG | 1 | 0 | 1 |
| Non-Talim black patch (wrong_dimensions) | 1 | 3 | 4 |
| **Total actual files** | **50** | **147** | **197** |

There is one additional missing-PNG record, not an image. There are ZERO confirmed
genuine invalid-Talim-notation images. There are 9 unusable/non-Talim artifact
files (4 white blanks + 1 corrupt + 4 black-patch variants), which must not be
reported as 9 independently authored semantic Talim negatives. All share the
same source family. The four white inputs are exactly constant255 and the
clean black input is exactly constant0 at 2x2. The three black variants are26x26
patches with padding/augmentation, not a Talim page.

The earlier audit incorrectly described wrong_dimensions as potentially resized
valid content. The generator and pixels show it replaces the page with black.
It also left scanned blanks unadjudicated; all three are actually constant white.
The prior unsupported-representation label was inherited by RGB-derived JPEGs,
although their Talim content remains valid. These are diagnostic corrections
only; no frozen labels or official metrics have been rewritten.

The original summary's invalid denominator2 is its historical blank/corrupt
label scope, not a complete inventory of the9 artifact files. All5 observed
REJECT_INVALID actions are correctly explained by input checks: 4 no-ink white
images and1 corrupt PNG. Three additional wrong-tier reviews are from the black
patch's count-only OCR X/Z. There is no valid expected count transcription for
these patches; do not blame a nonexistent ground-truth symbol or call them
confirmed tier swaps. The clean black image and missing image are the other2
fixture reviews. Valid-image syntax can still be wrong due to recognition.

## Weighting, uncertainty and policy interpretation

Negative fixtures: 198 records yield13 ACCEPT,180 REVIEW and5 REJECT. Ordinary
controls:236 yield229 ACCEPT and7 REVIEW. Of all187 reviews,180 contain the
wrong-tier code; 178 of those are fixtures and2 ordinary positives. The remaining
reviews include failed segmentation/recognition checks. See the case CSV for
every affected image, its original annotation, expected/predicted text, tier,
glyph, duplicate group, parent and attribution. Mutation and parent tables avoid
mistaking variants for independent evidence.

Main source of the wrong-tier pattern: role-blind recognition representation,
not confirmed erroneous D1 tier assignment. Cropping/normalization removes useful
page context, but saved row assignments match source roles on177 valid-content
wrong-tier cases. Exact crop/component preservation for every scanned variant
cannot be independently proven from saved boxes alone; processed crop pixels
were not archived, and no preprocessing replay was authorized/performed here.
The report therefore does not claim a full segmentation-fidelity proof.

Talim membership/count/canonical rules remain correct. F1 v2 correctly sends
OCR-only violations to REVIEW; this diagnosis gives no basis to revert them to
image-level invalidity or to tune the exposed evaluation. D1/E1, F1/F2 policies,
thresholds, datasets, manifests and historical results were left unchanged.
