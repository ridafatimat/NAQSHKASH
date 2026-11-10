# R3 invalid transcription fidelity audit
Date: 2026-11-07. Diagnostic analysis of exposed development observations; no new inference or decision evaluation.

## Inputs and reference eligibility

Discovered and verified completed runs:
- R0: C:\NAQSHKASH\stage2\stage2f_rejection_evaluation_v2\metrics\20261106_204802_280_R0
- R1: C:\NAQSHKASH\stage2\stage2f_rejection_evaluation_v2\metrics\20261106_204802_280_R1
- R2: C:\NAQSHKASH\stage2\stage2f_rejection_evaluation_v2\metrics\20261107_010212_401_R2
- Shared raw observations: C:\NAQSHKASH\stage2\stage2f_rejection_evaluation_v2\metrics\20261106_204802_280_raw

The existing development-only loader selected 1,400 INVALID images from 700 independent parents. References come exclusively from each mutated image's raw_display and verified layout, not original_structured_notation, original valid-parent labels or desired outcomes. All 1,400 have AUTO_VERIFIED structured display evidence; no reference is missing. The construction audit is fixed-font automated evidence, not expert review. Only the wrong-tier subset described in the companion report was freshly checked against pixels.

Three distinct measurements prevent a fabricated invalid CTC target:
1. **Visible ink-glyph fidelity:** remove ASCII spaces from each tier independently; retain row/tier identities and visible glyph order. Allow only the previously verified exact B/X equivalence at source font size 48. This is a diagnostic projection of visible glyphs, not whitespace transcription accuracy or structural validity.
2. **Tier-string pattern match:** trim blank canvas margins but preserve internal blanks. This is structured-display fidelity, not proof that character indices align across independently decoded tiers. No spatial CER is computed.
3. **Lexical exact/CER:** score visible ink-glyph streams only where a single character identity is supported. Conservatively exclude whole-page references containing any member of the existing near/exact alias exclusion groups B/X, C/Y, E/Z, G/[, I/backslash, J/], K/^, L/_. This leaves 508 potentially unambiguous pages: 498 wrong-tier, 2 count-order and 8 missing-count. Eight have no usable OCR, leaving 500 pages for lexical metrics. Count-only visible glyph streams are separately unambiguous for all categories; exclude 18 no-OCR pages, leaving 1,382. None is saved as a CTC target.

Missing-count notation has a legitimate visible-ink sequence but no complete plain-text representation of an absent cell's position. Reversed count glyphs have legitimate text/order, but symbol-to-token boundaries require geometry. Wrong-tier notation can be expressed with explicit tier labels when its glyph is unambiguous. Internal symbol spacing and empty regions remain structural data, not literal CTC space targets. The 892 alias-containing whole-page references are excluded from strict lexical CER, rather than assigning a convenient character label.

## Measured fidelity

Counts are images / distinct parents. Parent counts describe at least one affected variant and are not additive across conditions.

| Category | Total | Raw and T1 ink-glyph faithful | Raw and T1 tier-pattern faithful | Raw glyph discrepancy or unavailable OCR |
|---|---:|---:|---:|---:|
| invalid_count_token | 444 / 222 | 423 / 212 (95.27%) | 407 / 204 (91.67%) | 21 / 11 |
| missing_count | 458 / 229 | 424 / 212 (92.58%) | 112 / 56 (24.45%) | 34 / 17 |
| wrong_tier | 498 / 249 | 446 / 223 (89.56%) | 432 / 216 (86.75%) | 52 / 26 |
| ALL | 1400 / 700 | 1293 / 647 (92.36%) | 951 / 476 (67.93%) | 107 / 54 |

Raw literal tier-pattern matches are 943 / 472 parents; T1 literal matches are 951 / 476. Once B/X equivalence is respected, raw and T1 both match 951 patterns and 1,293 glyph streams. T1 changes label spelling for a known equivalent shape; it does not recover lost structural evidence. No output or decision was changed.

Of the 107 glyph-discrepant/unavailable cases, 18 / 9 parents have no OCR after D1 uncertainty; 89 / 45 parents have available output with a visible-glyph discrepancy. Separately, 216 / 108 missing-count images have a provable string-pattern loss: the predicted count equals the supplied visible count line with its internal blank(s) removed. The union of available-output glyph discrepancies and proven gap loss is **303 / 152 parents (21.64%)**, with two images overlapping both conditions. This union overlaps the 1,293 glyph-faithful cases: removal of a blank can preserve every ink glyph.

Among R2's 788 unsafe accepts, 739 / 370 parents retain all visible glyph identities/order, and 507 / 254 retain the complete tier-string pattern up to B/X equivalence. The latter comprise 407 count-order and 100 missing-count images. Thus OCR error alone cannot explain unsafe acceptance. A valid-looking F1 alphabet check is not a complete canonical parse.

## Reproducible primary causes and overlapping mechanisms

The diagnostic script applies the following precedence to each image, with secondary flags retained:
- **D, unscorable:** D1/OCR unavailable; or all-minimum-edit alignments disagree about the mutation's recognized character(s). Here all 18 D cases are unavailable OCR.
- **B, loss/change:** every minimum-edit alignment proves at least one mutated visible character changed/omitted, or an internal count gap demonstrably collapsed while its surrounding glyph sequence was retained. Edit alignment is a text-correspondence diagnostic, not measured image coordinates.
- **A, mutation faithful but unchecked:** the mutated visible character(s) survive all minimum-edit alignments, yet R2 ACCEPTS. These are count-order cases; missing-count absence is not assigned a fictitious character.
- **C, spatial-only primary:** the remaining missing-count cases need a coordinate/occupancy representation; text alone cannot assess the absent token.
- **Mutation preserved and REVIEWED:** a separate non-failure bucket, rather than falsely labeling a conservative REVIEW as a missed violation or an ambiguous glyph. Other, unrelated OCR discrepancies may still be secondary flags.

For count-order and missing-count, **C is also a secondary structural flag on all 902 images / 451 parents**, including A/B/D cases. Whole-page glyph discrepancies, gap loss, reference ambiguity, raw/T1 pattern match and paired-output identity are secondary flags. Primary B therefore differs from the broader 303-image evidence-loss union; some discrepancies concern unrelated glyphs.

| Category | A images/parents (%) | B images/parents (%) | C images/parents (%) | D images/parents (%) | Mutation preserved, REVIEWED (%) |
|---|---:|---:|---:|---:|---:|
| invalid_count_token | 429 / 215 (96.62%) | 3 / 2 (0.68%) | 0 / 0 (0.00%) | 0 / 0 (0.00%) | 12 / 6 (2.70%) |
| missing_count | 0 / 0 (0.00%) | 216 / 108 (47.16%) | 232 / 116 (50.66%) | 10 / 5 (2.18%) | 0 / 0 (0.00%) |
| wrong_tier | 0 / 0 (0.00%) | 38 / 19 (7.63%) | 0 / 0 (0.00%) | 8 / 4 (1.61%) | 452 / 226 (90.76%) |
| ALL | 429 / 215 (30.64%) | 257 / 129 (18.36%) | 232 / 116 (16.57%) | 18 / 9 (1.29%) | 464 / 232 (33.14%) |

Percentages use category image denominators. The five image buckets sum to 1,400. Parent buckets can overlap when clean/illumination variants differ; their sum is not an independent sample count.

The 429 A cases include six images with an affected-tier discrepancy elsewhere but the reversed token itself preserved. This is why local mutation fidelity exceeds the previously measured 423 accepted cases with an entirely faithful affected count tier. Three count-order images / two parents have a changed mutation token; the other 441 retain it. All 38 unsafe wrong-tier accepts change the mutated glyph; 452 wrong-tier cases preserve it and receive REVIEW, while eight lack OCR.

## Lexical metrics on legitimate visible-ink references only

Raw E1 and T1 are identical on every metric below. Exact means all scored row/tier glyph streams match; CER sums existing edit distance over those streams. Spaces and spatial absence are outside this metric. Denominators exclude unavailable OCR. Small eligible page subsets are not representative estimates for an entire category.

| Reference scope | Pages | Reference ink characters | Exact | CER |
|---|---:|---:|---:|---:|
| count_only_all | 1382 | 59226 | 1339 (96.89%) | 97 / 59226 (0.16%) |
| count_only_invalid_count_token | 444 | 18774 | 431 (97.07%) | 29 / 18774 (0.15%) |
| all | 500 | 42492 | 456 (91.20%) | 74 / 42492 (0.17%) |
| wrong_tier | 490 | 42346 | 446 (91.02%) | 74 / 42346 (0.17%) |
| count_only_wrong_tier | 490 | 22634 | 484 (98.78%) | 24 / 22634 (0.11%) |
| count_only_missing_count | 448 | 17818 | 424 (94.64%) | 44 / 17818 (0.25%) |
| missing_count | 8 | 110 | 8 (100.00%) | 0 / 110 (0.00%) |
| invalid_count_token | 2 | 36 | 2 (100.00%) | 0 / 36 (0.00%) |

These are newly permitted offline visible-glyph diagnostics. Historical invalid-image CER fields remain null and unchanged. A missing-count image may have zero ink-glyph CER while still lacking a required count.

## Representative mechanisms and pairing

- A: v2p_0065af0e59d893a535fa824d_invalid_count_token. Reversed hs survives in hsncrdca; F1/R2 accepts legal count characters without observed symbol/token boundaries.
- B: v2p_019d5d64a15fdc4a559d6c0c_missing_count. Display jajifka aa becomes jajifkaaa. The count is not supplied by a F1 default; a blank distinction is lost before validation.
- B: v2p_2b95f81713e613befddece0b_wrong_tier. Actual lower-tier D is decoded as permitted lower _. The companion pixel audit excludes exact/known conservative glyph equivalence.
- C: v2p_0321415d4f3aa36653d11710_missing_count. Absence needs source coordinates; no invisible count character is invented for scoring.
- D: v2p_160562f99ba3d2118cd03413_wrong_tier and its illumination variant have no usable OCR; their empty output is not an E1 transcription error.

The manifest's matched_valid_control_id points to the clean anchor even on illumination variants. This audit instead selects the valid control from the same parent with an identical augmentation recipe. Complete raw OCR is identical in only eight invalid/control pairs / four parents, all unavailable-OCR missing-count reviews. No successfully recognized unsafe-ACCEPT pair has identical complete OCR. This does not restore character positions or token associations.

Clean versus illumination: glyph fidelity is 647/700 versus 646/700; tier-pattern fidelity is 476/700 versus 475/700. These are repeated variants of the same 700 parents, not 1,400 independent sources.

## Reproduction and preservation

One optional script, src/r3_fidelity_diagnostic.py, prints statistics to stdout. Default is saved-observation/metadata analysis only. Add --cases for per-image primary causes and secondary flags, or --wrong-tier-pixels for the bounded 38-negative/38-control source-pixel audit. It creates no result files and runs no model.

~~~powershell
$R = 'C:\NAQSHKASH\stage2\stage2f_rejection_evaluation_v2'
$Python312 = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
$env:PYTHONPATH = "$R\src;C:\NAQSHKASH\stage2\positive_edge_cases\test_dependencies"
& $Python312 -B "$R\src\r3_fidelity_diagnostic.py"
~~~

No manual execution is required to obtain the measured findings above. No original decisions, reference fields, policy files, models or historical integrity locks were modified. Original 483-file history, 18-file experiment, R2 and 2,844-file preservation locks passed. Source result hashes:
- C:\NAQSHKASH\stage2\stage2f_rejection_evaluation_v2\metrics\20261106_204802_280_R0\results.json: af2310db1d6bbd4ebfa9c2e5ae9fb2dfb28d017e5253ecc0262c872e243052d0
- C:\NAQSHKASH\stage2\stage2f_rejection_evaluation_v2\metrics\20261106_204802_280_R1\results.json: 281fb036f110963b22da016fc8678ab873eaec114d8cd043d284cc4ed83c555d
- C:\NAQSHKASH\stage2\stage2f_rejection_evaluation_v2\metrics\20261107_010212_401_R2\results.json: 3d9c5cb814628a15abff858cabb03ce26919ed3b347af020d3bc56e5ca3ee2af

