# R3_HITL_V3 development report — template

Status: NOT EXECUTED. Populate from a completed manual run; do not copy Stage 1 counts into R3 results.

Run ID / timestamp: [pending]
R2, Stage 1, dataset and source/config/protocol hashes: [run_manifest.json]
Images / independent parents: [verify 2,800 / 700]
Policy semantics: at least one supported complete canonical page reading required; incomplete/unsupported evidence -> REVIEW; uniqueness not required; existing R2 REVIEW/REJECT preserved.

| Candidate | Valid ACCEPT | Valid REVIEW | Valid REJECT | Invalid ACCEPT | Invalid REVIEW | Invalid REJECT |
|---|---:|---:|---:|---:|---:|---:|
| R2 historical reference | 1376 | 24 | 0 | 788 | 612 | 0 |
| Stage 1 historical reference | 1318 | 82 | 0 | 48 | 1352 | 0 |
| R3_HITL_V3 actual run | pending | pending | pending | pending | pending | pending |

Required actual rates and denominators: invalid ACCEPT, valid definitive false REJECT, valid REVIEW, invalid REVIEW, overall REVIEW [pending]. REVIEW is not definitive rejection.

Invariants: original reviews/rejects preserved [pending]; new reviews [pending]; new definitive rejects [must be zero]; all image-specific violations [pending/none].

Stage 1 differences: list all IDs/reasons from Stage1_differences.csv. Uniqueness is not required. Review any differences due to fail-safe incomplete/unsupported evidence without forcing historical counts [pending].

Coverage/reasons: complete, capped, unavailable and unsupported searches; row/page ambiguity; all REVIEW reasons [pending].

Category/parent/clean-illumination/OCR/transcription results: [metrics.json]. Include independent-parent counts rather than treating variants as independent.

Localization: correct/eligible logical rows, tiers, comparable candidate columns; coarse source-region overlap; unknown/incorrect cases; conditional and full-denominator metrics; per-parent results [pending]. Broad overlap does not imply precise glyph localization. Exact pixel-location coverage is zero by design. Valid-review explanation/region availability is separate.

Optional gallery: selected IDs, geometry source/proof, generated versus text-only entries, approximate labels [pending]. No annotation-derived runtime rectangles.

Unresolved invalid accepts, unnecessary valid reviews and representative evidence: [pending]. No claim of autonomous invalid-image rejection.

Decision before freeze: [approve candidate / identify implementation defect / revise development protocol]. Freeze manifest and approval: [not prepared]. Validation and protected-test: [not authorized/executed].
