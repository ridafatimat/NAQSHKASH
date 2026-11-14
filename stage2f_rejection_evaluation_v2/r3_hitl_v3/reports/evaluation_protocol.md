# Evaluation protocol, schemas and freeze procedure

## Runtime contract

policy_v3.decide(r2_decision, interpreted_rows, availability, config=None, alignment_evidence=None)

- r2_decision: existing outcome in ACCEPT/REVIEW_UNCERTAIN/REJECT_INVALID; reason_codes list; optional reason_details list/explanation/retained evidence. R3 does not reinterpret an old rejection or review.
- interpreted_rows: list of dictionaries with only upper/count/lower strings, exactly the R2/T1 interpreted output. No annotation strings or repaired spaces.
- availability: preprocessing_status, segmentation_accepted (boolean), ocr_status. Other source metadata is not consulted.
- config: policy_id R3_HITL_V3, supported_complete_canonical_page acceptance requirement, REVIEW-only text action, monotonic invariants, Stage 1 computational budgets and dependency hashes. Confidence thresholds and unknown config fields are rejected.
- alignment_evidence: Stage 1 analysis, engine hash, parameters and hash of exact runtime text/status inputs. If absent, compute with the unchanged Stage 1 engine. The development evaluator may wrap saved analysis only after checking completed-run hashes, matching R2/raw provenance, exact development coverage, engine hash and budgets. Evidence from a different input is rejected.

Output contains policy_id, copied final decision, original R2 outcome, new_review flag, findings and alignment evidence. image_invalidity_newly_confirmed is always false. All originals are preserved. A noncanonical text candidate is never independently invalid image evidence.

At least one adequately supported complete canonical page interpretation permits retention of R2 ACCEPT; uniqueness is not required. Incomplete/unsupported evidence and validator uncertainty still receive REVIEW. Stage 1 counts are comparisons, not forced targets. No confidence threshold is introduced.

## Finding/region schema

Each new review has at least one finding: reason_code, stage, evidence_source, logical_row_index, human_readable_row_number, tier, candidate_text_columns and one-based display columns, location_status, observed_token/scope, associated_symbol, explanation, text_model_search_exhaustive, image_geometry_verified, recommended_reviewer_action, candidate offsets/failure evidence and separate region arrays.

Unknown fields are null/empty, not guessed. observed_token is the entire decoded count tier unless a different scope is explicitly supported. Candidate columns are sampled reconstructed-grid positions from canonical failure evidence. They are not source coordinates. Global total conflicts carry candidate rows and inferred totals, not a false culprit.

The policy's pixel-region arrays remain empty. After its decision, a separate region planner may use verified D1 metadata to produce processed boxes and source polygons/boxes. Region records carry dimensions, approximate row/tier scope and precise_invalid_glyph_verified=false. No renderer geometry participates. Overlay code reads selected source bytes only after the decision run is sealed, validates identity/dimensions and draws annotated thumbnails; absent mappings produce text only.

## Offline scoring contract

The runtime decision is completed before passing dataset validity/mutation data to localization_metrics.score. Only newly reviewed INVALID cases enter the invalid localization denominator. A matching logical-row count/order convention is required for row/tier scoring. Candidate text-column membership is eligible only when the observed count string exactly matches the source count sequence at known count-origin zero. No physical position is derived from that eligibility.

For mapped regions, compare broad source rectangles against source mutation CELL regions reconstructed offline from the supplied mutation/layout record. Wrong-tier source and destination cells may both be included. This is broad-region overlap, not exact changed-ink or glyph localization. Precise pixel localization has zero eligibility. Report eligibility, conditional accuracy, full-denominator recall, unknown/incorrect cases and parent-level metrics. Multiple highlighted rows/tiers are distinguished from uniquely correct identification. Valid REVIEW cases have availability metrics only.

## Development run protocol

Use the unchanged DevelopmentDataset and completed R2/Stage 1 run checkers. No original dataset images are read by evaluation. R3_HITL_V3, R2 and Stage 1 share the identical source IDs, interpreted transcription and saved raw observations. No label, mutation, expected count or generator layout enters R3.

Required review: all three confusion categories and rates; invalid accepts; valid definitive false rejects; overall/valid/invalid reviews; transitions; preserved R2 reviews; new R3 reviews; zero new definitive rejects; complete/incomplete/unsupported searches; reason counts; category/parent/clean-illumination/OCR/transcription results; localization eligibility/accuracy; all Stage 1 differences. An invalid REVIEW is not a definitive rejection. A valid REVIEW is a review burden, not a false definitive reject.

Only a new collision-safe directory under r3_hitl_v3/runs is writable for each evaluation. completion.json seals exact hashes after original dependencies, implementation and historical sources are checked again. Optional galleries have separate manifests and do not alter the sealed decision artifacts. No new integrity manifest depends on ZIP archives, temporary review repositories or directory-wide copies.

## Freeze preparation — not automatic freezing

After actual development execution and review, run the following in the same PowerShell session as the README. Do not run it merely because unit tests passed:

```powershell
& $Python312 -B "$R\src\evaluate_v3.py" prepare-freeze --run-dir $Out --ack-development-reviewed
if ($LASTEXITCODE -ne 0) { throw 'Freeze candidate generation failed.' }
Get-ChildItem "$R\runs\freeze_candidates"
```

This checks the completed run, monotonic invariants, zero newly introduced rejects, dependency hashes, exact evaluated source/config/protocol/test hashes and a passed-test record bound to the current source/config/test files. It writes a unique CANDIDATE_MANIFEST_NOT_AN_APPROVED_FREEZE JSON file. It does not mark the policy frozen, unlock a split, run tests/inference, or authorize held-out evaluation.

The manifest covers R3 source/config, Stage 1 engine, canonical validator/rules and external count/symbol definitions, relevant D1/E1/T1/F1/R2/config/vocabulary hashes, test code/results, evaluation protocol and exact schemas. No model is copied. Make a separate approval record with reviewer identity/date, candidate-manifest SHA and reviewed development-result SHA. Preserve the immutable manifest and approval together. The approved code/config must not be tuned using validation or test outcomes.

If development reveals a defect, preserve that run, correct the implementation, rerun tests, regenerate test_results.json from the actual test result/current source hashes, rerun development, explain the change and prepare a new candidate. Do not copy the old passed-test proof for changed code. Do not mutate a completed run or weaken a historical lock. Formatting/filling the FYP report template is separate from the evaluated source/protocol fingerprint.

## Held-out implementation remaining — currently blocked by design

The supplied CLI supports development_design only. The existing development loader continues to reject validation/protected selections. No hidden bypass, alternate split parameter or held-out execution command exists.

Before held-out execution, separately authorize and implement an isolated adapter that:
1. Reads only the explicitly authorized 600-image split and its 150 independent parents; verifies immutable split membership, parent/image/semantic disjointness and the dataset seal. It must not relax the development loader or scan other split image directories.
2. Produces/loads correctly lineage-bound frozen D1/E1/T1/R2 observations once for that split if they do not already exist. That inference is not implemented/executed here. Source labels/layout/mutations remain scoring-only, except already approved non-label rendering context in the original frozen pipeline; no new generator shortcut may enter R3.
3. Calls the approved, unchanged R3 API and exactly the frozen Stage 1 dependency. Captures D1 transform metadata for optional source overlays rather than guessing identity transforms.
4. Keeps decisions before mutation scoring and image overlays, uses legitimate valid-only transcription references, preserves protected evidence and logs access/run identities.
5. Is itself tested, audited and hash-frozen in an evaluation-adapter manifest referencing the approved R3 policy manifest before any held-out image access.

### Validation

With explicit authorization after development approval/code freeze, evaluate the 600-image validation split once. Report outcome counts/rates, invalid acceptance, valid definitive false rejection, valid/overall review, category/parent results, transcription limitations and localization coverage/accuracy versus development. Preserve the run. Do not silently tune after viewing results. A material defect requires transparent protocol/version handling; it does not justify opening the protected test early.

### Protected test

After validation review and separate authorization, evaluate the 600-image protected split once using the already selected frozen policy and frozen held-out adapter. Do not modify the policy using test outcomes. Report declines honestly and preserve all three decisions, including unresolved reviews. Further research requires a new protocol and appropriately reserved data, not reuse as a fresh test.

No held-out results exist for R3_HITL_V3 at implementation time. Freeze preparation and this document grant no authorization to read held-out images.
