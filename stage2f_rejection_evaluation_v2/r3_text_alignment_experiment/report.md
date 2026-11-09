# R3 Stage 1 — implementation status

**Full development evaluation NOT RUN. No measured Stage 1 improvement is claimed.**

Created the isolated text-only candidate enumerator, saved-R2 development evaluator, unit tests and README. The existing canonical_row and validate_aligned_row rules remain unchanged. The existing development loader, frozen lineage checks and R3 offline-fidelity helpers are reused.

The actual contract is literal OCR characters and spaces in the label-style leading-stripped representation, plus added leading offsets. Observed leading blanks are recorded unchanged but marked unsupported because their relation to crop padding is unproven; they are not silently normalized or interpreted as source geometry. Canonical-format necessity prunes the count offset to zero after translation within the supported model; it does not recover a physical crop anchor. Both upper/lower origin branches are covered. Natural offset bounds come solely from the observed count width and symbol-tier lengths. Default resource caps are 128 added columns and 258 candidate trials/row; capped or uncertain searches are explicitly incomplete/unscorable. Original internal and edge spaces are never silently repaired. No expected_cells or source-derived offsets are supplied.

Successful candidates give inferred run totals. Multiple-row analysis intersects those candidate-only totals, records remaining ambiguity and counts compatible page combinations. Completeness is conditional on unchanged text and prefix-only reconstruction; source-image geometry is always unknown. A contradiction in this model is not proof of an invalid image.

**23 lightweight synthetic tests passed.** They cover valid upper/lower and multi-count rows, alternative leading offsets, empty optional tiers, count reversal, first/middle/last missing counts, wrong-tier glyphs, ambiguous alignments, space collapse in both directions, repeated characters, conflicting row totals, unavailable OCR/D1, resource caps, canonical uncertainty, no expected-cell/ground-truth leakage and a small mocked offline comparison. Some known-invalid geometries admit passing text reconstructions; a valid source with collapsed OCR spaces can yield a complete literal-text contradiction. These are limitations, not test failures.

All 2,800 saved development observations can be included by the manual evaluator, including unsupported/incomplete cases. Outputs will show original R2 versus a separately named REVIEW-only diagnostic counterfactual, validity/category/variant/parent/OCR/fidelity breakdowns, and the historical 788/507/429 targeted cohorts. No expected remaining-ACCEPT count is imposed.

Whether this is useful for count-order errors, which retained missing-count patterns are recoverable, the false-REVIEW burden on valid controls, and the next Stage 2 design depend on those actual measurements. No new image validity claim, production rejection policy, OCR inference or tuning is implemented. See README.md for exact commands; real metrics will be written into a new runs/<timestamp>/report.md after manual execution.

Existing D1/E1/T1/F1/R2 code, checkpoint, datasets, historical metrics and integrity locks were not modified. No validation/protected-test image bytes were accessed. No Git operation occurred.
