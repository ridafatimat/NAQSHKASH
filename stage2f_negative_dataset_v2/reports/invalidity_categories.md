# Approved categories and exclusions

Three content categories are supported, each by rerendered image evidence and
an independent inverse canonical parser. The preflight target is10 parents per
category, one clean invalid + one matched valid per parent, with one illumination
variant of each. Parents vary independently of category in rows, symbols, counts
and width; dense short-run pages cover longer sequences as well.

1. **wrong_tier** / CANON_SYMBOL_DIRECTION: move one supported glyph from its
   legal symbol tier into the opposite tier at the same horizontal run position.
   The target must have no legal same-raster or conservatively audited lookalike; B/X and any detected alias
   are excluded. Both actual glyph locations change, without changing counts.
2. **invalid_count_token** / CANON_COUNT_ENCODING: reverse a two-glyph authentic
   token only if the reversed token is outside the authentic259-token set.
   Example ja -> aj. Preserve one symbol run and the visible cell locations.
   The independent verifier tries alternative one/two-cell segmentations; a
   string-level 'not in set' check alone is not the admission criterion.
3. **missing_count** / CANON_REQUIRED_SYMBOL_COUNT: erase an entire visible count
   token while retaining its symbol and surrounding instructions. Both token
   glyphs are removed for a two-cell count. No reflow or inferred replacement is
   allowed. Any surviving valid canonical alternate interpretation excludes it.

Exclusions: JSON-only count/type/schema/direction changes; duplicate IDs or
metadata failures; TXT-only changes; B/X alias relabelings; range260/300 encoded
with unsupported/fallback '?' glyphs; count changes yielding another valid count;
spacing/alignment conventions needing domain expertise; partial glyph erasures
that merely form another valid symbol; arbitrary extra tiers/layouts not proved
invalid; unknown or foreign notation. UNSUPPORTED is different from INVALID.

Blank/corrupt/unreadable inputs are a separate IMAGE_QUALITY_FAILURE task, not
invalid Talim semantics. They are not generated to fill this dataset's quota.
Similarly, current D1/E1/F1 errors never supply labels or admission decisions.

Illumination variants reuse the existing scan engine with a mild0.92/0.96
brightness factor and lossless PNG. The same recipe is applied to both matched
classes. The verifier independently transfers every atlas cell through the
documented brightness LUT and reruns ALL grammar alternatives. This certifies
evidence retention without learning/tuning a confidence threshold. It is a narrow
illumination-style collection, not a broad realistic scan/photo benchmark.
JPEG, blur, noise, rotation, perspective, resolution and shadows are deferred
until independent verification can account for their geometry/losses. No claim
is made that coarse contrast alone preserves semantic invalidity.

Statuses: AUTO_VERIFIED = code/raster proof in the audited scope, not human
review; NEEDS_DOMAIN_REVIEW = no sufficient automatic proof; EXCLUDED = unchanged,
alternate-valid, inconsistent or otherwise unproven. The generator stops on a
fundamental control/render/admission failure and never fills a quota with an
unverified category. Expert review of the chosen canonical scope remains welcome.
