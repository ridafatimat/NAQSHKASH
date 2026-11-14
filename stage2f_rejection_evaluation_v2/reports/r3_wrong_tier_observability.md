# R3 wrong-tier observability audit
Date: 2026-11-07. Scope: all 38 R1 unsafe wrong-tier accepts, 19 independent development parents.

## Method and measured result

The optional diagnostic inspected and hash-checked the 38 negative source images and 38 **same-recipe** valid controls (76 distinct image files). The manifest's matched_valid_control_id is a clean anchor; same-illumination controls were selected from the existing parent quartet, avoiding brightness-induced whole-page differences.

Using the existing independent_verify module offline, the audit recovered actual raster cells under the documented source48 font/grid and pointwise brightness conditions, searched canonical interpretations, and compared the moved glyph against every permitted symbol at the observed target tier. It reused the verified T1 font audit and the more conservative dataset admission alias groups. No font matching or generator-layout rule was introduced into runtime validation.

Measured:
- **0/38** moved glyphs have an exact same-tier permitted equivalent after visible-ink trimming, in either clean or illumination rendering.
- **0/38** have a permitted alternative in the existing conservative alias groups or inverse-renderer cell candidates.
- **38/38** independently raster-parse as INVALID with CANON_SYMBOL_DIRECTION, with no legal canonical interpretation under this source-rendering contract.
- **38/38** have a uniquely matched actual moved glyph; all have nonzero localized source pixel changes against the same-recipe valid control.
- **38/38** replace/change the mutated glyph in the saved OCR correspondence; none is merely a B/X spelling ambiguity. The counts are D->lower: 16 images / 8 parents; U->upper: 10 / 5; F->lower: 8 / 4; A->lower: 2 / 1; T->upper: 2 / 1.
- R1 and R2 ACCEPT all 38. T1 does not restore these distinct glyph identities. All 38 have accepted D1 proposals and no reported perspective correction; the saved records do not establish identity deskew or complete component preservation.

The exact-ink comparison includes crop-to-visible-ink translation, not just full-cell byte equality. The separate conservative audit covers near-equivalent marks such as C/Y, E/Z and L/_, even where their rasters differ. None of the five moved symbols is in those alias groups. Source-size48 B/X equivalence therefore cannot explain these accepts.

## Parent-level findings

Each row represents the clean image ID formed by appending _wrong_tier to the parent and its _wrong_tier_illumination variant. Both variants have the same original symbol, target tier and local recognized replacement. OCR correspondences use all minimum-edit alignments; every row has a unique replacement, not a conveniently chosen alignment.

| Source parent | Actual moved glyph | Target tier | OCR replacement | Exact or audited conservative legal alias |
|---|---|---|---|---|
| v2p_2b95f81713e613befddece0b | D | lower | _ | none |
| v2p_373c7dac9f4d529616dcb496 | D | lower | _ | none |
| v2p_3bd818f9bb2a9e4ac95b1764 | U | upper | C | none |
| v2p_46470fdda9900199375f4c9f | A | lower | _ | none |
| v2p_46f9a29731a1034b84184f06 | F | lower | Z | none |
| v2p_49224a1b7f2fd25c5950b4a0 | F | lower | Z | none |
| v2p_4b12b029685c7d963d5eb30b | U | upper | C | none |
| v2p_68beea701e9b9af01fbb2032 | D | lower | _ | none |
| v2p_75ec06927fe4ad4348ddedbf | D | lower | _ | none |
| v2p_7b099a95619cdd691807ffa1 | U | upper | C | none |
| v2p_8f3d82cc45b5d16bc79f8551 | D | lower | _ | none |
| v2p_a9c14fa214121094622541c1 | D | lower | _ | none |
| v2p_bbe37ee0114e3b36e302d9fa | D | lower | _ | none |
| v2p_bce3ef51a554e16eef518aca | U | upper | C | none |
| v2p_bde3f35b7ffb597fb8df5c33 | F | lower | Z | none |
| v2p_c0011cb1b338fe201c9d8d4d | U | upper | C | none |
| v2p_c833bf3129532f19d065608d | F | lower | Z | none |
| v2p_cac0058bdf26df9ea4feb8bc | D | lower | _ | none |
| v2p_f107c8ac52c9c98c3569532b | T | upper | E | none |

Matched controls retain the original symbol in its legal tier, while the invalid image moves that same mark into the opposite tier. For clean and same-recipe illumination variants, changed-pixel counts are 682 for D, 458 for U, 548 for F, 418 for A and 552 for T. These counts verify the manipulation but are **not** a perceptual distinguishability score.

## Visual similarity and limitations

A compact transient font comparison was inspected, without delivering a new gallery. D is two dots plus one bar, whereas its visually close lower-tier _ is a two-bar mark; U's three-dot mark differs from upper C's dot configuration; F is a dot/bar mark; A is a three-dot mark; T is a bar/dot mark. The nearest legal masks can share strokes or coarse structure, but the distinguishing dot/bar components are visible at the audited original rendering. Existing 32x32 shape distances were used only to choose comparisons, not as equivalence thresholds or calibrated confidence.

**No intrinsically invisible invalidity was found within the verified source-font/brightness model.** This is narrower than proving reliable human perception under every viewing condition. There is no expert-rated just-noticeable-difference study, and no validated near-similarity cutoff. Human/domain review remains necessary before treating automated source-font evidence as a general rejection guarantee, especially for resized/degraded imagery.

The source-image versus OCR disagreement is confirmed. Precisely assigning every lost dot/bar to D1 photometric/crop processing, resizing or the E1 recognizer is **not** possible from the saved cache: processed tier pixels, frame paths and full transforms are missing. Thus these are confirmed D1/E1 recognition-path failures rather than intrinsically equivalent source notation; do not claim all 38 are isolated neural-model errors. A future bounded extraction should retain processed crops and compare distinguishing components before and after resize.

The zero-alternative result is scoped to documented canonical rules and the known font, explicitly including existing aliases. It does not establish that an arbitrary upload uses the same renderer. The inverse renderer is an offline dataset audit tool and must not become a synthetic-grid runtime classifier. No labels or images were changed, and no validation/protected image content was accessed.

