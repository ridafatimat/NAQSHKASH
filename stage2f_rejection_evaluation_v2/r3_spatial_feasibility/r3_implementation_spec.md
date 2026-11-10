# R3 implementation specification — recommendation only
Date: 2026-11-07. No full R3 or new production policy implemented.

## Evidence-driven direction

Preserve frozen D1/E1/T1/F1/R2 and Stage 1. Keep the existing Stage 1 result as a diagnostic counterfactual, not a deployed rejector. Use text contradictions to request corroboration, and use image observations to test whether a plausible transcription is actually printed.

Current evidence: 50/70 valid contradictions are spacing-only, eight new reviews involve count-character errors, and all 48 residual invalid accepts have a plausible text witness. CTC coordinates are mathematically consistent but their spike timing is not reliable localization. An offline true-position control solves valid associations, while forced nonempty groups fail on eight missing-count cases. These findings favor **image-derived positions plus a separate glyph-identity/readability check**, rather than a fitted CTC offset or weaker Talim grammar.

## Proposed future three-way contract

| Decision | Necessary support |
|---|---|
| ACCEPT | Adequate readable-image coverage, reliable row/tier roles, no unexplained foreground/crop loss, image-supported glyph identities (including scoped alias handling), and consistent symbol/count associations/totals under the unchanged canonical rules. A text witness alone is insufficient. A spacing contradiction may be resolved only by independently recovered positions, not annotation-driven re-spacing or changing characters to force a pass. |
| REVIEW_UNCERTAIN | OCR-only contradiction, unresolved image/text disagreement, uncertain glyph/group boundaries, legal alternative glyphs/associations, unknown rendering, incomplete component coverage, suspicious empty corridor, unreliable crop/segmentation, or unvalidated spatial evidence. Retain REVIEW whenever a plausible image-supported valid alternative remains. |
| REJECT_INVALID | A readable, sufficiently observed image whose independently corroborated glyph/position evidence leaves no supported canonical interpretation. The finding must be localized, reproducible and robust to plausible legal glyph/group/association alternatives. No CTC confidence-only or blank-only rejection is allowed. |

Category-specific requirements:
- **Wrong tier:** reliable tier role plus independent visible glyph identity incompatible with every permitted glyph under that tier/rendering. Component presence/count and an OCR guess do not establish identity. Exact B/X and conservative near-alias cases fail the definitive-rejection gate unless resolved independently.
- **Count order:** independently identified visible count glyph sequence plus actual symbol-run boundaries showing noncanonical ordering, with alternative canonical assignments excluded. A different offset witness must be checked against the image, not simply ignored.
- **Missing count:** a verified symbol/run and readable, fully captured expected count region with independent image support for absent count ink, excluding crop loss, occlusion, fragmented grouping or an alternative run association. CTC blanks and an empty projection corridor alone are insufficient.
- Existing independent unreadable/no-ink artifact checks remain unchanged. This specification adds no new grammar.

CTC score and CTC span timing are outputs of the same model and are not independent confirmations. D1 geometry does not identify glyphs. Pixel projections locate foreground but do not prove a valid glyph interpretation or trustworthy absence.

## Minimal implementation boundary for a later authorized phase

1. Retain immutable raw text, Stage 1 candidate sets and reason provenance.
2. Add a label-blind image-evidence record: processed-coordinate group alternatives, component completeness, crop/readability status and source transforms.
3. Keep multiple grouping/identity/association interpretations where unresolved; do not pick whichever agrees with a label or expected count.
4. Apply the existing canonical checker only to supported reconstructions. Record whether all image-supported alternatives fail.
5. Permit a definite rejection only through the independent-image gate above; otherwise REVIEW. Do not promote all Stage 1 contradictions to REJECT.
6. Measure recovered valid ACCEPT, unsafe invalid ACCEPT, new valid REVIEW and false valid REJECT separately, with parent-level denominators and clean/illumination consistency.

This task does not implement those runtime steps. The only implemented image method is a diagnostic baseline and cannot currently satisfy the rejection gate.

## Exact manual VS Code commands

The target text/coordinate audit was already performed from saved records. The **48-case cached-component measurement requires manual execution**. It reads cached processed crops only: no source dataset image bytes, OCR, D1 pass or training.

~~~powershell
Set-Location 'C:\NAQSHKASH'
$S = 'C:\NAQSHKASH\stage2\stage2f_rejection_evaluation_v2\r3_spatial_feasibility'
$Python312 = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
$env:PYTHONPATH = "$S\src;C:\NAQSHKASH\stage2\stage2f_rejection_evaluation_v2\src;C:\NAQSHKASH\stage2\positive_edge_cases\test_dependencies"
$Out = "$S\runs\$(Get-Date -Format 'yyyyMMdd_HHmmss_fff')_components"

& $Python312 -B "$S\src\spatial_diagnostic.py" components --output-dir $Out
if ($LASTEXITCODE -ne 0) { throw 'Cached component diagnostic failed.' }

Get-Content "$Out\report.md"
Write-Host "Metrics: $Out\metrics.json"
Write-Host "Per-image grouping evidence: $Out\per_image.json"
~~~

Optional reproduction of the saved text/coordinate audit (stdout JSON, no new inference):

~~~powershell
& $Python312 -B "$S\src\spatial_diagnostic.py" audit
if ($LASTEXITCODE -ne 0) { throw 'Saved-artifact audit failed.' }
~~~

Lightweight tests only:

~~~powershell
& $Python312 -B -m unittest discover -s "$S\tests" -v
if ($LASTEXITCODE -ne 0) { throw 'Spatial diagnostic tests failed.' }
~~~

The script discovers latest completed runs, verifies completion hashes and selected development identities, and creates only a new output directory below its own runs folder. No resume or overwriting of previous experiments is attempted. Invalid inputs/hash drift abort instead of silently switching sources.

## Next decision and execution boundary

First inspect the manual component results for grouping coverage, valid-control false-empty/wrong-association risk and empty-token observability. Do not select/tune a threshold on these exposed outcomes. Next obtain independently scoped crop evidence for the uncached 70-valid and remaining-invalid cohorts, without rerunning OCR if D1-only extraction suffices. No reduction in their reviews or definitive rejection performance is established by the current cache.

Completed here: saved-reference analysis of all 70/48 cases, mapping/code audit, offline oracle association control, five reports, a small offline component diagnostic and ten synthetic tests. Not executed: the real cached-component batch, any coordinate correction/rescore, new source-image processing, inference, retraining or a full R3 decision pipeline. No validation/protected images, Git operations or historical changes occurred.

## Inputs and preservation

Stage 1: C:\NAQSHKASH\stage2\stage2f_rejection_evaluation_v2\r3_text_alignment_experiment\runs\20261107_172939_188
CTC evaluation: C:\NAQSHKASH\stage2\stage2f_rejection_evaluation_v2\r3_alignment_experiment\runs\20261107_021046_514_evaluation
CTC extraction: C:\NAQSHKASH\stage2\stage2f_rejection_evaluation_v2\r3_alignment_experiment\runs\20261107_021046_514_extraction

Source hashes, captured before/after analysis (all historical inputs are read-only):
- C:\NAQSHKASH\stage2\stage2f_rejection_evaluation_v2\r3_text_alignment_experiment\runs\20261107_172939_188\completion.json: 90b7d0c576e1bd4c6c09ba79813632d021c044cca4e15295f4a9895d5fac750a
- C:\NAQSHKASH\stage2\stage2f_rejection_evaluation_v2\r3_text_alignment_experiment\runs\20261107_172939_188\metrics.json: 1a72cc71290252e52eabcdfa26c0fa22faea7051c55952abb18325365f4b6347
- C:\NAQSHKASH\stage2\stage2f_rejection_evaluation_v2\r3_text_alignment_experiment\runs\20261107_172939_188\report.md: 1ac190488c9bf30bbccf78a9fb1ed085939ca6aef11d186b86c645aef2293891
- C:\NAQSHKASH\stage2\stage2f_rejection_evaluation_v2\r3_text_alignment_experiment\runs\20261107_172939_188\per_image.jsonl: 1f1b396a4b5e4e6d91a12d179e27a1dff5c311df5e3c5ab1f5e7bbfa424c12a6
- C:\NAQSHKASH\stage2\stage2f_rejection_evaluation_v2\metrics\20261107_010212_401_R2\results.json: 3d9c5cb814628a15abff858cabb03ce26919ed3b347af020d3bc56e5ca3ee2af
- C:\NAQSHKASH\stage2\stage2f_rejection_evaluation_v2\r3_alignment_experiment\runs\20261107_021046_514_evaluation\completion.json: 5729f4d6d49684037a7163af8fa0b858003cb514d34dc729ac0ccc411fdfb66b
- C:\NAQSHKASH\stage2\stage2f_rejection_evaluation_v2\r3_alignment_experiment\runs\20261107_021046_514_evaluation\metrics.json: bebbfdee8974bf6ac5cb4a1399b849ac846fdedda59c36c35a2f5913fa743b5f
- C:\NAQSHKASH\stage2\stage2f_rejection_evaluation_v2\r3_alignment_experiment\runs\20261107_021046_514_evaluation\report.md: f087a759eacc166b49a8005971b6c9c34c2bbbffd3e73b221320d8cef4565ea4
- C:\NAQSHKASH\stage2\stage2f_rejection_evaluation_v2\r3_alignment_experiment\runs\20261107_021046_514_evaluation\per_image.json: 3e3372057850be6dec2372913a4361ccbb60128f26494ca2f3a35383fb47a2a2
- C:\NAQSHKASH\stage2\stage2f_rejection_evaluation_v2\r3_alignment_experiment\runs\20261107_021046_514_extraction\completion.json: 6178b567301459e1c102b69badff9e2b7fc3f647f41b764ca6c70f507ef43654
- C:\NAQSHKASH\stage2\stage2f_rejection_evaluation_v2\r3_alignment_experiment\runs\20261107_021046_514_extraction\plan.json: d0eae4701c89ed6fcb060068be7ff59c00c9df93c10a675956c04e9dd33366fc
- C:\NAQSHKASH\stage2\stage2f_rejection_evaluation_v2\r3_alignment_experiment\runs\20261107_021046_514_extraction\observations.jsonl: 3d7adae75f7324268274834e077631faa2ad14dbe637aa5b416810502db800fa
