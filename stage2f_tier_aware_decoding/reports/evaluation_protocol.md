# Development-only protocol

T0: original saved E1 greedy rows -> unchanged F1 v2. T1: same raw rows ->
audited contextual interpretation -> unchanged canonical validator/F1 v2 plus
explicit unresolved-context guard. All raw rows, reasons/evidence and corrected
rows are retained. Independent input-artifact rejection remains strongest.

Only the exact saved 776-image development baseline at
C:\NAQSHKASH\stage2\stage2f_negative_rejection\metrics\20261106_154728_812_dev_baseline
is supported. CLI rejects alternate paths, including the exposed434 evaluation.
These development controls were previously used for training/selection and are
not independent test evidence. All variants are reported with original parent
IDs; macro-parent rates accompany image-weighted rates. Mapping selection uses
font/canonical role evidence, never evaluation outcomes. No full T0/T1 evaluation
was executed during implementation.

T1 uses only B/X equivalence in its audited rendering scope and uniquely legal
tier role. Reliable role means existing D1 acceptance with consistent saved
row/role/crop metadata/path, a sole legal alternative or an existing margin at
least the original D1 configured threshold. No additional threshold is selected.
Known synthetic sources must supply matching rendering metadata. Unknown font,
size32, missing/ambiguous role, conflict, unknown character or multiple targets
blocks correction. Unresolved tiers are not partly repaired. Source labels and
transcription_exact/expected outcomes are not runtime correction inputs.

Exact page transcription compares complete row/role dictionaries, preserving
meaningful internal spaces. CER sums Levenshtein edits over (row index,tier role)
keys divided by total ground-truth tier characters. Missing tiers charge all
missing characters; extra tiers charge insertions. It does not discard rejected
or segmentation-failed pages. Row/role alignment is explicit; leading blank
offsets follow existing E1 target conventions and cannot verify original
horizontal alignment. Zero denominators are null, not perfect scores.

Accepted-output correctness = exact interpreted pages / accepted scorable pages.
End-to-end correctness = ACCEPT and exact interpreted page / all scorable pages.
Review rate = REVIEW / all images. Wrong-tier review count retains the original
reason code on the candidate decision; raw wrong-tier decisions remain in every
row. Syntax-failure-resolved is descriptive and is not transcription correctness.

Every changed character is evaluated against ALL minimum-edit alignments between
raw tier and truth tier. It is scored correct/incorrect only if those alignments
give one unambiguous truth character at that raw position. Deletion/ambiguous
alignments are reported unscorable. Worsened page edit counts are also reported.
This avoids claiming success from changing REVIEW to ACCEPT or selecting a
favorable alignment. No corrections are silently removed from error analysis.

By-cohort outputs include origins, long tiers (maximum truth-tier length>=65,
following existing Stage2E descriptive length convention) and multi-logical-row
pages. These are evaluation-only descriptive cohorts, never decoding thresholds.
They do not claim to exhaust all forms of Talim complexity. Parent macro averages
weight parents equally; CER's image/global aggregate weights truth characters.

T0/T1 comparisons require identical raw baseline hashes, IDs/parents/truth,
mapping and experiment code. Output folders are new and completion-hashed.
Comparison generates actual metrics and error galleries only from completed
runs. All changed cases are retained in JSON/CSV, while gallery image display
is bounded by --gallery-limit. No trained checkpoints or simulated performance
results are created. New independent data or model changes require separate
authorization after reviewing the development experiment.
