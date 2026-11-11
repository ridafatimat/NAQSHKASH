# Glyph ambiguity audit

All65 fixed vocabulary characters (space included), 2080 pairs, were compared at
24,32,48,64,96 source font pixels using the local Talamat.ttf, grayscale L masks,
integer positions and the project's lt anchor convention. Visible ink is trimmed
for equivalence because D1 produces ink-band crops; full font whitespace differs.
The renderer's actual source size48 is confirmed in original Stage1 and positive
source JSON rendering metadata. E1 target-height64 is a resized crop, not proof
that source font size was64. No font file was copied.

Exact visible pairs: size24 B/X and C/Y; size32 none; size48 B/X; size64 B/X and
C/Y; size96 B/X. Thus no pair is claimed byte-identical under every rendering.
Only B/X is exact at the audited project source size48. B is upper-only and X
lower-only; a reliable upper/lower role uniquely resolves that label ambiguity.
The verified mapping is two directed entries. Default T1 scope is original
source size48, L mode, this font hash and integer lt rendering. Uploaded images
without trusted rendering provenance remain uncertain; source32 is blocked.
Scanned variants inherit scope only from verified source48 metadata, not from
an assumed observed pixel size. Equality before the same scan transformation
does not establish that segmentation or the whole transcription is correct.

C/Y, [ / G and _ / L are merely similar at48 and are excluded. Normalized shape
distances rank comparison illustrations only; they are neither calibrated scores
nor correction thresholds. Pairs sharing the same role cannot be resolved from
tier identity alone. Count-tier symbols are drawn from the canonical encoded
1..259 values; no unsupported notation or numeric constraints are invented.

Every correction still requires reliable D1 metadata, one legal target, absence
of conflicting evidence and no remaining unsupported character in that tier.
This is contextual interpretation of a discrete label, not a new CTC decoding
algorithm or proof that source image content is valid. No evaluation labels or
exposed434 outcomes were consulted to construct the mapping.
