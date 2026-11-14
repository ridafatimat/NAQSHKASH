# Frozen development-only evaluation protocol

Ground-truth classes are VALID and canonical-rule-verified INVALID. The output
classes remain ACCEPT, REJECT_INVALID and REVIEW_UNCERTAIN. Unknown/unverified
source examples are rejected by this initial dataset loader rather than relabeled.
All commands allow ONLY development_design. Other split arguments are rejected
before image/annotation access; no allow-test override exists.

Invalid rejection rate = invalid REJECT_INVALID / invalid images. Invalid accepted
rate = invalid ACCEPT / invalid images. Invalid review rate = invalid REVIEW /
invalid images. These three sum to1. REVIEW is NOT a successful definitive
rejection. Also report how many invalid inputs pass INTERPRETED SYNTAX regardless
of the final action; syntax-pass does not imply correct/valid input content.

Valid acceptance = valid ACCEPT / valid images. False definitive rejection =
valid REJECT_INVALID / valid images. Valid review = valid REVIEW / valid images.
Exact and CER are computed ONLY for valid inputs with legitimate references.
References retain row/role structure, strip only leading tier blanks as E1 did,
and preserve internal spaces. CER sums per-(row,tier) Levenshtein edits / truth
characters, charging deleted/extra tiers; failed segmentations are not dropped.
Invalid-image CER and exact references are null, never computed against original
valid parents or fabricated valid CTC targets. Their evaluation is semantic
three-way behavior, category/rule evidence and structural-trigger diagnostics.

Accepted-output correctness = exact accepted VALID outputs / accepted scorable
VALID outputs. An accepted INVALID output is instead a critical semantic failure;
it is not mixed into a misleading CER denominator. Valid end-to-end correctness
requires ACCEPT AND exact transcription / all scorable valid pages. Overall
expected semantic-action accuracy requires the expected three-way action;
transcription remains separately measured. Empty denominators are null.

T1 character correctness uses the existing all-minimum-edit alignment algorithm
on valid references: unique expected character -> correct/incorrect edit; ambiguous
alignment/deletion -> unscorable. ALL invalid-image interventions are unscorable
as transcription edits without a legitimate CTC reference; their false-accept/
review/reject consequences are still fully measured. Page edit-count worsening
is also reported. No success is claimed from REVIEW -> ACCEPT alone.

Strata: original invalidity category; paired mutation category (matched controls
included); clean/illuminated; short maximum visible-tier length<=4, medium5..64,
long>=65; single/multi-row. Length is descriptive ground-truth metadata only,
never a threshold or correction input. Group counts overlap across strata.
Parent macro rates weight eligible parents equally. Per-parent worst cases flag
ANY invalid acceptance, ANY valid definitive rejection, ANY accepted wrong valid
transcription and ANY unresolved review. Severity is a reporting convention,
not a new runtime decision policy.

Returned preprocessing errors, D1 proposal failures, OCR status failures and
unlocalized pipeline exceptions have separate counts/lists. Original F1 reason
codes and evidence survive R1 interpretation. Reason frequencies count images
per code; expected canonical rule IDs are not misrepresented as detected rules.

R0/R1 must have identical dataset selection, raw file hash, raw plan, checkpoint,
record IDs/parents, raw predictions and references. Comparison reports critical
regressions and transcription metrics without choosing/tuning a winner. Inspect
errors even if both candidates perform poorly. No thresholds or model/policy
parameters change automatically. Failure galleries are display-limited, but full
failure CSV/JSON lists preserve all selected records and never include protected
or validation images.
