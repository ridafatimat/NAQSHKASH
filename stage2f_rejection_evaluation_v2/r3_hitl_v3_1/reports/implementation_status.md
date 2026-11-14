# Implementation status

R3_HITL_V3_1 is a separate implementation candidate. Historical R3 source, configuration, notes and completed development artifacts are not edited. Read-only hashes cover the specified R3 development run and the reused dependencies. No Git/GitHub operations, OCR, retraining, vocabulary changes or held-out-image access are part of this work.

The D1-only adapter and synthetic policy/measurement/evaluation tests are implemented. The full development evaluation, reference offset accuracy, claimed ambiguity resolution, valid-review impact and reduction of residual invalid ACCEPT remain unmeasured here and must be verified by the user's manual run. No freeze is prepared automatically.

Generation is blocked for unresolved ambiguity and for unique text-only ACCEPT without verified positioning. Every retained ACCEPT has an explicit selected-reading status and ordered runs. Existing R3 review regions, findings and offline localization results are preserved; no precise new glyph overlay is asserted.

Unresolved measurement limitations: count projection fragments/merges; agreement with erroneous OCR; limited pitch support in short examples; differing glyph bearings; perspective/nonuniform cell pitch; and an uncertainty interval crossing the fixed 0.5-column tolerance. These produce unavailable evidence when detected. No source-font template, generator geometry, source mutation or expected offsets are supplied to runtime positioning.

See README.md for exact manual commands and runtime/scoring schemas, and test_results.json for the final executed lightweight test counts and implementation hashes.

Final lightweight verification: **59 passed / 0 failed**. Metadata-only preflight matched 2,800 development records / 700 parents to the specified completed R3 run, reading no image pixels.
