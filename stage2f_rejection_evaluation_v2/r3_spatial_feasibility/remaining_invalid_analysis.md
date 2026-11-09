# Remaining invalid ACCEPT analysis
Date: 2026-11-07. Scope: all 48 residual Stage 1 accepts / 24 independent development parents.

## Measured cases

| Category | Images | Parents | Mechanism |
|---|---:|---:|---|
| wrong_tier | 32 | 16 | The source wrong-tier glyph is decoded as a permitted same-tier glyph |
| invalid_count_token | 8 | 4 | A faithful reversed-token string admits a different canonical offset/token grouping |
| missing_count | 8 | 4 | Six trailing omissions and two collapsed internal omissions admit a valid-looking text reconstruction |

All 48 have at least one saved canonical page witness with a consistent inferred total. These witnesses use the recognized strings, not the true source geometry. Thus none is distinguishable as definitively invalid from **every possible valid interpretation of these strings** using the current text model. This does not mean the images or matched controls are identical.

All eight remaining count-order images preserve the visible tier patterns. Their 1-2 passing page candidates reassign symbol/count relationships; adding another text-only count-order rule would reject one interpretation without establishing the printed grouping. Six missing-count cases retain their edge-trimmed tier patterns, while two internal-gap cases lose the blank. These source omissions cannot be inferred from a CTC blank or a shorter count string alone.

The 32 wrong-tier substitutions are D->_, U->C, F->Z, A->_ or T->E at their observed target tiers. The prior source-font audit found no exact or conservative permitted glyph alias for these moved marks. That source evidence is distinct from what the runtime OCR knows. A valid-looking substitute erases the wrong-tier violation from the text; offsets cannot recover the original glyph identity.

## Matched valid controls and cache limits

Each case was compared offline with the **same parent and same illumination recipe** valid control. None of the 48 complete interpreted transcriptions is identical to its matched control. Yet all have a canonical text alternative, so difference from a chosen control is not an invalidity criterion. Controls, labels and generator parameters were not used to produce image groups, candidate spatial associations or decisions.

Only **10 cases / five parents** are in the CTC cache: eight wrong-tier images / four parents plus two trailing-missing-count images / one parent. None of the eight residual count-order cases is cached. The broader cache contains other count-order pilot cases, which are useful method examples but do not establish success on these residual cases.

## Complete residual inventory

Both clean and illumination variants are listed. For wrong-tier rows, evidence is source glyph -> saved OCR replacement from all minimum-edit correspondences. For missing-count rows, location is classified from the visible mutated display offline.

| Image ID | Category | Source/OCR or omission location | Passing page candidates | Cached crops |
|---|---|---|---:|---|
| v2p_031616149474aee0abbf6b29_invalid_count_token | invalid_count_token | internal | 1 | no |
| v2p_031616149474aee0abbf6b29_invalid_count_token_illumination | invalid_count_token | internal | 1 | no |
| v2p_1179dc85af988bd51ecd0ec0_missing_count | missing_count | trailing | 1 | no |
| v2p_1179dc85af988bd51ecd0ec0_missing_count_illumination | missing_count | trailing | 1 | no |
| v2p_2b95f81713e613befddece0b_wrong_tier | wrong_tier | D -> _ | 1 | no |
| v2p_2b95f81713e613befddece0b_wrong_tier_illumination | wrong_tier | D -> _ | 1 | no |
| v2p_343d8433ff172be99eaf0309_invalid_count_token | invalid_count_token | internal | 1 | no |
| v2p_343d8433ff172be99eaf0309_invalid_count_token_illumination | invalid_count_token | internal | 1 | no |
| v2p_373c7dac9f4d529616dcb496_wrong_tier | wrong_tier | D -> _ | 1 | no |
| v2p_373c7dac9f4d529616dcb496_wrong_tier_illumination | wrong_tier | D -> _ | 1 | no |
| v2p_46470fdda9900199375f4c9f_wrong_tier | wrong_tier | A -> _ | 1 | no |
| v2p_46470fdda9900199375f4c9f_wrong_tier_illumination | wrong_tier | A -> _ | 1 | no |
| v2p_46f9a29731a1034b84184f06_wrong_tier | wrong_tier | F -> Z | 1 | no |
| v2p_46f9a29731a1034b84184f06_wrong_tier_illumination | wrong_tier | F -> Z | 1 | no |
| v2p_4b12b029685c7d963d5eb30b_wrong_tier | wrong_tier | U -> C | 2 | yes |
| v2p_4b12b029685c7d963d5eb30b_wrong_tier_illumination | wrong_tier | U -> C | 2 | yes |
| v2p_57c37cdc2432c8e3828c292d_missing_count | missing_count | trailing | 1 | no |
| v2p_57c37cdc2432c8e3828c292d_missing_count_illumination | missing_count | trailing | 1 | no |
| v2p_5aaa54f4af316a958b3e7561_invalid_count_token | invalid_count_token | internal | 1 | no |
| v2p_5aaa54f4af316a958b3e7561_invalid_count_token_illumination | invalid_count_token | internal | 1 | no |
| v2p_5e3c8423e42c0ce864f59c58_missing_count | missing_count | internal | 3 | no |
| v2p_5e3c8423e42c0ce864f59c58_missing_count_illumination | missing_count | internal | 3 | no |
| v2p_668eeb63b6bdfb61ba44f67d_invalid_count_token | invalid_count_token | internal | 2 | no |
| v2p_668eeb63b6bdfb61ba44f67d_invalid_count_token_illumination | invalid_count_token | internal | 2 | no |
| v2p_68beea701e9b9af01fbb2032_wrong_tier | wrong_tier | D -> _ | 1 | yes |
| v2p_68beea701e9b9af01fbb2032_wrong_tier_illumination | wrong_tier | D -> _ | 1 | yes |
| v2p_75ec06927fe4ad4348ddedbf_wrong_tier | wrong_tier | D -> _ | 1 | no |
| v2p_75ec06927fe4ad4348ddedbf_wrong_tier_illumination | wrong_tier | D -> _ | 1 | no |
| v2p_7b099a95619cdd691807ffa1_wrong_tier | wrong_tier | U -> C | 1 | no |
| v2p_7b099a95619cdd691807ffa1_wrong_tier_illumination | wrong_tier | U -> C | 1 | no |
| v2p_8f3d82cc45b5d16bc79f8551_wrong_tier | wrong_tier | D -> _ | 1 | no |
| v2p_8f3d82cc45b5d16bc79f8551_wrong_tier_illumination | wrong_tier | D -> _ | 1 | no |
| v2p_a9c14fa214121094622541c1_wrong_tier | wrong_tier | D -> _ | 1 | no |
| v2p_a9c14fa214121094622541c1_wrong_tier_illumination | wrong_tier | D -> _ | 1 | no |
| v2p_adaa99947b071410ea7d8cbd_missing_count | missing_count | trailing | 2 | yes |
| v2p_adaa99947b071410ea7d8cbd_missing_count_illumination | missing_count | trailing | 2 | yes |
| v2p_bbe37ee0114e3b36e302d9fa_wrong_tier | wrong_tier | D -> _ | 1 | no |
| v2p_bbe37ee0114e3b36e302d9fa_wrong_tier_illumination | wrong_tier | D -> _ | 1 | no |
| v2p_bde3f35b7ffb597fb8df5c33_wrong_tier | wrong_tier | F -> Z | 1 | no |
| v2p_bde3f35b7ffb597fb8df5c33_wrong_tier_illumination | wrong_tier | F -> Z | 1 | no |
| v2p_c0011cb1b338fe201c9d8d4d_wrong_tier | wrong_tier | U -> C | 1 | no |
| v2p_c0011cb1b338fe201c9d8d4d_wrong_tier_illumination | wrong_tier | U -> C | 1 | no |
| v2p_c833bf3129532f19d065608d_wrong_tier | wrong_tier | F -> Z | 1 | yes |
| v2p_c833bf3129532f19d065608d_wrong_tier_illumination | wrong_tier | F -> Z | 1 | yes |
| v2p_cac0058bdf26df9ea4feb8bc_wrong_tier | wrong_tier | D -> _ | 1 | no |
| v2p_cac0058bdf26df9ea4feb8bc_wrong_tier_illumination | wrong_tier | D -> _ | 1 | no |
| v2p_f107c8ac52c9c98c3569532b_wrong_tier | wrong_tier | T -> E | 1 | yes |
| v2p_f107c8ac52c9c98c3569532b_wrong_tier_illumination | wrong_tier | T -> E | 1 | yes |

## Minimum additional evidence

- Wrong tier: independently establish the visible glyph identity, exclude legal visual aliases, and corroborate the D1 tier role. Ink positions/component counts alone are not semantic recognition.
- Count order: establish count glyph identities/order and the actual symbol-run boundaries. A source-renderer layout is an offline reference, not a runtime solution.
- Missing count: establish a reliable symbol/run region, readable count corridor and genuinely absent printed count, excluding crop loss, occlusion and ambiguous glyph grouping. CTC non-emission is insufficient.

Preserve REVIEW whenever an image-supported legal alternative remains. No residual case was newly rejected or relabeled.
