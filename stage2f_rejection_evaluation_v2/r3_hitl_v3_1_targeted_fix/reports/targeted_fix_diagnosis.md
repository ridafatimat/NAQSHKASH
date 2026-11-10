# Diagnosis of the failed R3.1 run

Read-only source: `r3_hitl_v3_1/runs/20261107_203523_866_d442571d`.

## Confirmed failure chain

- All 2,800 regenerated box records equal saved D1 observations (including records where D1 rejected and both saved/new row lists are empty).
- First ink is present on all 16,444 saved measurable tiers. Geometry and first-ink extraction are not the dominant failure.
- 2,730 images have the repeated-character shape veto alone; another 6 have it together with grouping-cardinality failure. Thus 2,736 hit that unsupported veto.
- 34 report inconsistent raw glyph-start spacing. Another 30 are existing D1 rejection cases (6 no legal layout, 18 unassigned foreground, 4 role ambiguity, 2 unresolved vertical merge).
- Pixel-exact shape equality across repeated OCR characters is unsupported. A decoded character may be wrong, and legitimate occurrences can have different binarized silhouettes. In the inspected first parent, the same decoded `a` has two silhouette hashes despite the same 35-by-29 bounding dimensions.
- Adjacent different glyphs have different bearings: observed starts include steps 57, 43, 51, 49 and 62 pixels. Those steps are not interchangeable with a cell-grid pitch.

## Bounded label-blind diagnostic evidence

Only three cached cases were examined with the bearing-cancelling calculation, not all 2,800 cases. No real D1/OCR inference or full development policy rescore was performed. An existing valid source image was visually inspected; source fields were used separately to diagnose the scoring schema, never to infer runtime pitch.

`v2p_0065af0e59d893a535fa824d_invalid_count_token` and `v2p_0065af0e59d893a535fa824d_valid` each have five successive same-silhouette translation observations, including 100/2, 350/7 and 150/3 pixels per decoded column. All yield 50 without source layout/labels. `v2p_0082651e8bf5bf8715736da9_valid` has only two observations (350/7), so the unchanged three-observation support requirement correctly leaves it unavailable. Fixture IDs identify provenance only; the estimator reads no validity/category field.

This supports a narrow correction of the existing estimator: use identical silhouette matches as POSITIVE evidence and cancel their bearing, rather than require all decoded-character occurrences to match or assume every different-glyph ink step is the grid pitch. No uncertainty threshold is increased. Short/unrepeated/fragmented/spacing-inconsistent inputs remain unscorable.

## Scoring defect

The old run reports 1,180 eligible valid ACCEPT selections, 0 correct and 1,180 false, despite zero generation-ready outputs. Those are unverified text-only selections, not verified incorrectly selected carpets. Moreover, source runs carry `direction` and `encoded_count` metadata while canonical candidates carry only `symbol` and `count`. Whole-dictionary equality falsely marks matching symbol/count runs unequal. The inspected selected and source first row of `v2p_0082651e8bf5bf8715736da9_valid` demonstrates this schema mismatch.

The targeted scorer separates unavailable selections (null correctness), eligibility/coverage and truly incorrect verified carpets. Ordered symbol/count pairs are compared without dropping any semantic symbol/count difference. The historical run is not rescored or overwritten.

## Boundary and stop condition

No assertion is made that this correction recovers all measurements, resolves 138 pages or reaches approximately 32 invalid ACCEPTs. Matching shape-group translations still depend on correct grouping and saved OCR column spacing. Physical first-ink offsets retain bearing uncertainty. The final manual run must expose failures rather than force counts. If reliable positioning remains unavailable or unsafe, retain verified R3_HITL_V3 and stop R3.1 work; no alternate architecture is proposed.
