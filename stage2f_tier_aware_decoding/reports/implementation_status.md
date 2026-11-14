# Implementation status

The isolated T0/T1 prototype, glyph/role audits, saved-development evaluation,
paired comparison, parent/cohort metrics and error-gallery tools are implemented.
No full candidate evaluation, CRNN inference, training, backward pass, optimizer
step, confidence calibration, threshold tuning or Git operation was executed.
No font file was copied. No fake model or performance results were created.

The mapping was built from all65 font characters and canonical roles at five
source sizes, not from the exposed434 outcomes. Only the project's verified
B/X context is enabled by default. Four saved DEVELOPMENT observations were
used for regression tests; the three other wrong-tier confusions remain blocked.

Focused tests cover upper/lower direction, unknown/ambiguous roles and glyphs,
multiple alternatives, raw/evidence preservation, ground-truth independence,
disabled-baseline identity, F1 v2 compatibility, long internal spacing,
incorrect/unscorable corrections, parent macro denominators, paired artifact
comparisons, gallery/CSV roundtrips and historical/experiment integrity.
See lightweight_tests.txt for the final test count and command outcome.

Historical inputs are protected by configs/historical_lock.json, and new
experiment code/config/mapping/protocol files by configs/experiment_lock.json.
These locks are frozen before candidate performance is measured. Rebuilding
them after results would require a separately reviewed experiment version.

Pending user execution: T0 and T1 full development postprocessing/evaluation,
actual CER/exact/accepted-correctness and correction-harm results, parent/long-
sequence breakdowns and the final error gallery. No improvement claim is made
from the unit fixtures. Unknown real fonts, non-equivalent glyph confusions and
uncertain segmentation need additional data/design work or future model context.
