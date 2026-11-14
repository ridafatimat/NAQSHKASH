# Valid contradiction analysis
Date: 2026-11-07. Read-only diagnosis of the latest completed Stage 1 run.

Source: C:\NAQSHKASH\stage2\stage2f_rejection_evaluation_v2\r3_text_alignment_experiment\runs\20261107_172939_188. Its metrics.json, report.md and all per_image.jsonl records were read; completion and evidence hashes were verified. Baseline R2 results were matched to the frozen development manifest. No Stage 1 evaluation was rerun.

## Confirmed findings

All **70 valid contradictions / 36 parents** were compared with their own AUTO_VERIFIED source display and saved interpreted OCR. All 70 original fully aligned source rows pass the unchanged validate_aligned_row; there are **zero confirmed canonical-rule failures on these valid source annotations**. This is a saved-reference audit, not a new manual pixel adjudication of the 70 images.

| Primary cause | All contradictions images / parents | New ACCEPT-to-REVIEW images / parents |
|---|---:|---:|
| Glyph sequence preserved; spacing differs | 50 / 26 | 50 / 26 |
| Count-character errors, sometimes also symbol errors | 12 / 6 | 8 / 4 |
| Symbol-character errors without count-character errors | 8 / 5 | 0 / 0 |
| Total | 70 / 36 | 58 / 30 |

Parent sets overlap: one parent has different clean/illumination failure types. The primary buckets partition images, not independent parents. Classification treats only the previously verified source48 B/X equivalence as an equivalent ink label, consistent with the existing fidelity audit.

Secondary observations:
- **52 images / 27 parents** have a space-pattern discrepancy somewhere: 50 spacing-only and two mixed count-error cases.
- **Two images / one parent** have passing row candidates but conflicting row totals. For v2p_17d4516de37e4f7072d1cd5f_valid and its illumination variant, inferred totals are [189], [229], [242], versus source totals [189,189,189]. This is an OCR-content problem causing a page-total contradiction; do not force totals back to 189.
- No observed leading/trailing whitespace occurs among these 70. Unsupported leading offsets are therefore **not their measured cause**. The separate 30 Stage 1 unscorable images must not be merged into this contradiction cohort.
- Every contradiction is pattern-unfaithful; none preserves the exact source tier pattern up to B/X. No case proves the canonical grammar should be weakened.
- NONCANONICAL_COUNT_TOKEN failures frequently arise because lost symbol-tier spaces move inferred run boundaries, not because the source count order is invalid. Failure reason codes alone do not establish a count-symbol error.

Example spacing-only discrepancy (lower, row 0):
- Source: 'W   SW S           W'
- OCR: 'W   SW S        W'

The visible glyph sequence is unchanged; only the spacing differs. A prefix-only offset search cannot restore internal gaps. This is an intended limitation of the literal reconstruction model, which is unnecessarily narrow as an image-validity inference when OCR loses spaces; it is not an unnecessarily restrictive canonical Talim rule.

## Remedies, with evidence boundaries

For the 50 spacing-only new reviews, investigate pixel-derived glyph/run positions independently of the CTC string's spaces. Do not respace from source annotations or accept whichever spacing yields a desired answer. For eight new count-error cases, geometry alone cannot repair the character identities; require an independent image-based check or retain REVIEW. The 12 pre-existing reviews remain unresolved by this diagnosis.

All 70 lack processed crops in the current 48-image cache: **0 images / 0 parents overlap**. Cached pilot controls therefore cannot demonstrate recovery of these 58 new reviews. A later, separately authorized D1-only crop capture on this exact cohort could supply spatial evidence without rerunning E1, but it was not implemented or executed here. No change to the current REVIEW assignments is supported by the cache alone.

## Complete image inventory

Spaces column counts row/tier string comparisons with identical glyph streams but different whitespace. “Total only” means every row has candidates but their totals do not intersect. Source image IDs are listed individually; source parents are the portion preceding _valid.

| Image ID | Primary cause | R2 transition | Space discrepancies | Total only |
|---|---|---|---:|---|
| v2p_041d172c5d2c1fb2203df4cc_valid | space pattern only | new | 1 | no |
| v2p_041d172c5d2c1fb2203df4cc_valid_illumination | space pattern only | new | 1 | no |
| v2p_16bbde963ac9d58465f41263_valid | space pattern only | new | 1 | no |
| v2p_16bbde963ac9d58465f41263_valid_illumination | space pattern only | new | 1 | no |
| v2p_17d4516de37e4f7072d1cd5f_valid | count glyph(s), possibly other errors | new | 0 | yes |
| v2p_17d4516de37e4f7072d1cd5f_valid_illumination | count glyph(s), possibly other errors | new | 0 | yes |
| v2p_1c52fd5a10d946a365c3f185_valid | symbol glyph(s) | already REVIEW | 0 | no |
| v2p_3480f7553dd17a915579c6df_valid | space pattern only | new | 1 | no |
| v2p_3480f7553dd17a915579c6df_valid_illumination | space pattern only | new | 1 | no |
| v2p_3e8b4a866318c10380e5c1aa_valid | symbol glyph(s) | already REVIEW | 0 | no |
| v2p_3e8b4a866318c10380e5c1aa_valid_illumination | symbol glyph(s) | already REVIEW | 0 | no |
| v2p_3ffd8ca85c51c8338e7a39cd_valid | count glyph(s), possibly other errors | new | 1 | no |
| v2p_3ffd8ca85c51c8338e7a39cd_valid_illumination | count glyph(s), possibly other errors | new | 1 | no |
| v2p_427fceb630f088f0720883a2_valid | space pattern only | new | 1 | no |
| v2p_427fceb630f088f0720883a2_valid_illumination | space pattern only | new | 1 | no |
| v2p_470aa0942ff1c346ddec6255_valid | space pattern only | new | 1 | no |
| v2p_49224a1b7f2fd25c5950b4a0_valid | space pattern only | new | 1 | no |
| v2p_49224a1b7f2fd25c5950b4a0_valid_illumination | space pattern only | new | 1 | no |
| v2p_511597f4e2d6605286556866_valid | space pattern only | new | 2 | no |
| v2p_511597f4e2d6605286556866_valid_illumination | space pattern only | new | 2 | no |
| v2p_5ed64ac9686273eb55606555_valid | space pattern only | new | 1 | no |
| v2p_5ed64ac9686273eb55606555_valid_illumination | space pattern only | new | 1 | no |
| v2p_5f10345844cd3319b1cf531c_valid | space pattern only | new | 1 | no |
| v2p_5f10345844cd3319b1cf531c_valid_illumination | space pattern only | new | 1 | no |
| v2p_619bb54d1ae30a45138da8fe_valid | space pattern only | new | 1 | no |
| v2p_619bb54d1ae30a45138da8fe_valid_illumination | space pattern only | new | 1 | no |
| v2p_6324fa336eb75f9ec55d76fb_valid | space pattern only | new | 1 | no |
| v2p_6324fa336eb75f9ec55d76fb_valid_illumination | space pattern only | new | 1 | no |
| v2p_7227f20b0e8dc527b8cf9f53_valid | count glyph(s), possibly other errors | already REVIEW | 0 | no |
| v2p_7227f20b0e8dc527b8cf9f53_valid_illumination | count glyph(s), possibly other errors | already REVIEW | 0 | no |
| v2p_722aead49a1ef07fbc137edf_valid | space pattern only | new | 1 | no |
| v2p_722aead49a1ef07fbc137edf_valid_illumination | space pattern only | new | 1 | no |
| v2p_76b302652d661ad2a13b8c7e_valid | space pattern only | new | 1 | no |
| v2p_76b302652d661ad2a13b8c7e_valid_illumination | space pattern only | new | 1 | no |
| v2p_7c6c1691197299678337880f_valid | space pattern only | new | 1 | no |
| v2p_7c6c1691197299678337880f_valid_illumination | space pattern only | new | 1 | no |
| v2p_7e3a3b6e06e7f459fba66567_valid | symbol glyph(s) | already REVIEW | 0 | no |
| v2p_7e3a3b6e06e7f459fba66567_valid_illumination | symbol glyph(s) | already REVIEW | 0 | no |
| v2p_7ef815a7e6473f0284ce39af_valid | count glyph(s), possibly other errors | new | 0 | no |
| v2p_7ef815a7e6473f0284ce39af_valid_illumination | count glyph(s), possibly other errors | new | 0 | no |
| v2p_868c3ad6830adda6f165e3ee_valid | count glyph(s), possibly other errors | new | 0 | no |
| v2p_868c3ad6830adda6f165e3ee_valid_illumination | count glyph(s), possibly other errors | new | 0 | no |
| v2p_86b1538744f552dc9d91f48a_valid | space pattern only | new | 1 | no |
| v2p_86b1538744f552dc9d91f48a_valid_illumination | space pattern only | new | 1 | no |
| v2p_919e90be196092eca29a443a_valid | space pattern only | new | 1 | no |
| v2p_919e90be196092eca29a443a_valid_illumination | space pattern only | new | 1 | no |
| v2p_95ee45581fa001295c55aea3_valid | space pattern only | new | 1 | no |
| v2p_95ee45581fa001295c55aea3_valid_illumination | space pattern only | new | 1 | no |
| v2p_970c2e28d7b97c1d5e2a252b_valid | count glyph(s), possibly other errors | already REVIEW | 0 | no |
| v2p_970c2e28d7b97c1d5e2a252b_valid_illumination | count glyph(s), possibly other errors | already REVIEW | 0 | no |
| v2p_b03c31fd7e55e7fbf16e570a_valid | space pattern only | new | 1 | no |
| v2p_b03c31fd7e55e7fbf16e570a_valid_illumination | space pattern only | new | 1 | no |
| v2p_c15e7478d0d30b4a2493b339_valid | space pattern only | new | 1 | no |
| v2p_c15e7478d0d30b4a2493b339_valid_illumination | space pattern only | new | 1 | no |
| v2p_c86b74d7099ed3437a68f351_valid | space pattern only | new | 1 | no |
| v2p_c86b74d7099ed3437a68f351_valid_illumination | space pattern only | new | 1 | no |
| v2p_c9e58b8acc79e701e77995ec_valid | space pattern only | new | 1 | no |
| v2p_c9e58b8acc79e701e77995ec_valid_illumination | space pattern only | new | 1 | no |
| v2p_d0810357e07752572b4defd8_valid | symbol glyph(s) | already REVIEW | 0 | no |
| v2p_d0810357e07752572b4defd8_valid_illumination | symbol glyph(s) | already REVIEW | 0 | no |
| v2p_e32d9079cc0fbdf47c033d68_valid | space pattern only | new | 1 | no |
| v2p_e32d9079cc0fbdf47c033d68_valid_illumination | symbol glyph(s) | already REVIEW | 0 | no |
| v2p_e928339fc721395f2ead5e39_valid | space pattern only | new | 1 | no |
| v2p_e928339fc721395f2ead5e39_valid_illumination | space pattern only | new | 1 | no |
| v2p_ebf30d21afb6c4a9c206d796_valid | space pattern only | new | 1 | no |
| v2p_ebf30d21afb6c4a9c206d796_valid_illumination | space pattern only | new | 1 | no |
| v2p_ec2ca284ef5fa2d5673aff1e_valid | space pattern only | new | 1 | no |
| v2p_ec2ca284ef5fa2d5673aff1e_valid_illumination | space pattern only | new | 1 | no |
| v2p_f503f3d09389a686f95381a3_valid | space pattern only | new | 1 | no |
| v2p_f503f3d09389a686f95381a3_valid_illumination | space pattern only | new | 1 | no |

Per-image source/predicted spacing snippets, failing reason counts and row totals are reproducible with the audit subcommand. Actual source annotation glyphs/counts enter only this offline diagnosis, never image grouping or future runtime offset decisions.
