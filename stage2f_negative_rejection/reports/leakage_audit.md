# Leakage audit

All negative records share one original embedded family. All clean/scanned
variants and duplicate IDs/semantics stay in evaluation. Variant IDs are not
independent source parents. Negative split is not a train/test experiment.

Historical positive assignment maps and Stage2C parent/page manifests were read.
Protected test parents are excluded; historical tuning/validation assignments
are preserved. Stage2E full-development model selection is explicitly recorded
as exposed, including clean regression. No new independent test is claimed.

Overlap checks: negative content byte hashes against the historical selected
page source/image hashes; source parent IDs are namespaced by data origin.
E1 training/evaluation tier manifests describe cropped tiers, so raw page hash
nonmatches cannot exclude semantic or near-duplicate overlap with training.
The Stage1 full dataset was NOT swept. Global semantic nonoverlap remains unknown;
use exposed-development interpretation even if selected-page hash overlaps are zero.
Complete historical lock covers integration code/config/manifests, source fixtures,
original Stage1 and E1 checkpoints; it is a scoped contract lock, not a claim that
every historical large ZIP/derived image was rehashed. No historical files are written.

Selected historical page byte-hash matches: []

Frozen manifests: 776 development pages/230 parents; 434 evaluation records/99 parents (including one negative family and one missing-image record). Cross-partition byte-identical image hashes: zero. This does not establish semantic nonoverlap.
