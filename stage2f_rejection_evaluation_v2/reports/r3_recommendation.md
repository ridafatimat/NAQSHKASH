# R3 recommendation
Date: 2026-11-07. Diagnostic conclusion only. No R3 implementation or alignment experiment was executed.

## Recommended minimum next step: Option B

Authorize a **small targeted CTC evidence-extraction feasibility pass**, as specified in r3_ctc_alignment_audit.md, before choosing any production R3 validator. The existing 2,800 observations have no frame paths/logits or complete source transforms. A second bounded pass is necessary to measure alignment; retraining is not necessary to expose these tensors.

Evidence motivating this choice:
- 1,293/1,400 invalid images preserve visible glyph identities/order, but only 951 preserve the tier-string pattern. Spatial association is a larger missing capability than aggregate glyph CER suggests.
- R2 still accepts 507 images / 254 parents with a faithful complete tier-string pattern; these are 407 count-order and 100 missing-count cases. More confidence thresholds cannot add absent geometry.
- 303 images / 152 parents have available-output glyph discrepancies or proven internal-gap loss; 18 / 9 lack OCR. These overlap other fidelity flags and do not form a single model-error estimate.
- The 38 unsafe wrong-tier images / 19 parents are distinct from permitted glyphs under the audited rendering, but their processed crops/paths are unavailable, so D1 versus E1 attribution is incomplete.

## Options ranked

| Rank | Option | Feasibility and evidence | Complexity and limitations |
|---:|---|---|---|
| 1 | B: targeted CTC extraction | Frozen model can expose logits/paths without retraining. Provides directly measured model-time evidence and transform bookkeeping. | Small isolated diagnostic pass. Localization quality is unknown; cannot recover unrecognized characters or prove absence from blank emissions. |
| 2 | C: image glyph/component positions | Potentially supplies independent visible-region support and missing-count occupancy evidence. | Disconnected dots/bars make component-to-glyph grouping nontrivial. Must work from image evidence and canonical rules, not fixed generator cell metadata. Needs its own bounded validation. |
| 3 | D: structural validation plus later E1 fine-tuning | Confirmed wrong glyphs and gap loss justify studying recognition errors after processed-crop diagnosis. | Training is not the minimum next action and is not authorized here. Alignment/structure does not follow automatically from lower CER. Any future training needs image-faithful references, alias handling and a separate protocol. |
| 4 | A: saved observations alone | Useful for the existing conservative gap REVIEW and post-hoc diagnostics. | Cannot reconstruct missing frame indices/transforms or reliably assign tokens to symbols. No meaningful evidence-backed definitive-rejection upgrade is supported now. |

Option B is a feasibility experiment, not a promise that CTC alone will solve R3. If associations fail its provisional gates, move to C. Missing-count absence will likely require C-like independent image support even if alignment is good. Do not count model confidence and its frame geometry as independent glyph evidence.

## What can safely change decisions later

- **Wrong tier:** current distinct/unsupported OCR glyphs can justify REVIEW. A valid-looking substitution cannot be rejected on the saved string alone. Definitive rejection might be supported by reliably localized, independently identified visible wrong-tier ink with all plausible legal glyph alternatives excluded. B/X and the conservative near-alias cases remain REVIEW unless independent evidence resolves them.
- **Count order:** preserved glyph order plus missing run boundaries currently supports investigation, not definite invalidity. A future reliable symbol/count association and independently corroborated visible token order could support applying the existing canonical validator. Multiple plausible associations require REVIEW.
- **Missing count:** internal decoded gaps already support R2 REVIEW. Shortened/edge-trimmed strings and CTC blanks cannot prove an absent count. Definitive rejection requires readable image regions, a supported symbol/run location and independently verified count absence under existing canonical rules.
- **D1/no OCR or unsupported representation:** retain REVIEW. Do not equate missing observations with missing printed notation.

The current experiments provide no measured frame-location error, token-association accuracy, alignment coverage or valid-control false-violation rate. No numerical acceptance cutoff or rejection-rate improvement is claimed. All provisional gates are documented in the alignment report.

## Execution status and reproducibility

Completed: saved development fidelity audit, all 38 wrong-tier pixel audits with same-recipe valid controls, source/checkpoint/schema inspection and the four reports. The single optional script src/r3_fidelity_diagnostic.py recomputes the measured statistics; --cases prints per-image primary/secondary flags, and --wrong-tier-pixels repeats the bounded source-pixel audit. **You do not need to run a script to complete this diagnostic.** No CTC extraction script or R3 implementation was created.

No CRNN inference, training, full R0/R1/R2 rerun, threshold tuning, dataset regeneration, source-policy modification, historical-lock rewrite or Git operation occurred. Checkpoint metadata was loaded for inspection only. Validation/protected image bytes were not accessed. Existing integrity locks passed before and after analysis. The only delivered additions are these four reports and the optional diagnostic script.

Proposed 12-parent/48-image alignment extraction remains **unexecuted and requires separate authorization**.

