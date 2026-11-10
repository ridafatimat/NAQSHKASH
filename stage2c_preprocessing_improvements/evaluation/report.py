"""Final Stage2C scientific report. Never changes selection or historical stages."""
import json,sys
from pathlib import Path
from collections import Counter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core import *

def percentage(value):return 'n/a' if value is None else f'{100*value:.3f}%'

def table(rows):
 return '\n'.join('| '+ ' | '.join(str(v) for v in row)+' |' for row in rows)

def main():
 root=HERE/'reports/experiments_v1';summary=json.loads((root/'summary.json').read_text());selection=json.loads((root/'selection_before_validation.json').read_text());completion=json.loads((root/'completion.json').read_text())
 tests=json.loads((HERE/'reports/tests.json').read_text());assert tests['failed']==0 and tests['errors']==0
 annotations=json.loads((HERE/'reports/annotation_correspondence.json').read_text());assert annotations['parents_checked']==478 and annotations['all_passed']
 lock=verify_lock(code=True);before=json.loads((HERE/'protocol/protected_before.json').read_text());assert protected()==before
 winner=selection['selected'];component=json.loads((root/'component_audit.json').read_text());design=json.loads((HERE/'protocol/design.json').read_text())
 pages=read_csv(root/'pages.csv');preds=read_csv(root/'predictions.csv');index={(r['partition'],r['dataset'],r['kind'],r['candidate']):r for r in summary}
 cohort_sections=[]
 for dataset in ('positive','general'):
  records=[index[('validation',dataset,'scanned',c)] for c in CANDIDATES]
  lines=[[r['candidate'],f"{r['proposals']}/{r['pages']}",f"{r['correct_pages']}/{r['pages']}",r['wrong_acceptances'],r['safe_rejections'],percentage(r['accepted_segmentation_accuracy']),r['tiers'],percentage(r['cer']),percentage(r['exact_accuracy']),percentage(r['page_exact_accuracy'])] for r in records]
  cohort_sections.append(f"### {dataset} internal validation — scanned pages\n\n| Candidate | Proposals | Correct segmentation | Wrong acceptances | Rejections | Correct among accepted | OCR tiers | Conditional CER | Exact tiers | Page exact/all |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"+table(lines))
  common=[[r['candidate'],r['common_pages'],r['common_tiers'],percentage(r['common_cer']),percentage(r['common_exact_accuracy'])] for r in records]
  cohort_sections.append('\n| Candidate | Common pages | Common tiers | Common CER | Common exact accuracy |\n|---|---:|---:|---:|---:|\n'+table(common))
 clean=[]
 for c in CANDIDATES:
  r=index[('clean_regression','normal','clean',c)]
  clean.append([c,f"{r['correct_pages']}/{r['pages']}",r['wrong_acceptances'],r['safe_rejections'],percentage(r['cer']),percentage(r['exact_accuracy']),r['common_pages'],percentage(r['common_cer'])])
 clean_table=table(clean)
 failure_lines=[]
 for c in CANDIDATES:
  reasons=Counter(r['reason'] for r in pages if r['candidate']==c and r['partition']=='validation' and r['kind']=='scanned' and r['correct_segmentation']=='False')
  failure_lines.append(f"- **{c}:** "+'; '.join(f'{reason}: {count}' for reason,count in reasons.most_common())+'.')
 boot=json.loads((root/'parent_bootstrap.json').read_text())
 uncertainty='\n'.join(f"- {r['dataset']}: correct scanned segmentation coverage change {100*r['mean_change']:+.2f} percentage points; parent-bootstrap 95% interval [{100*r['lower95']:+.2f}, {100*r['upper95']:+.2f}], {r['parent_count']} parents." for r in boot)
 component_table=table([[r['candidate'],r['pages'],r['structurally_aligned_pages'],r['components_evaluated'],r['components_with_loss'],r['pages_all_components_preserved'],r['aggregate_pass_pages_with_component_loss']] for r in component['candidates']])
 regression=index[('clean_regression','normal','clean',winner)];baseline=index[('clean_regression','normal','clean','C0')]
 wrong_total=sum(index[('validation',d,'scanned',winner)]['wrong_acceptances'] for d in ['positive','general'])
 recommendation=('Do not promote the selected configuration into the main Stage 1 workflow. It improves recall/coverage but still wrongly accepts '+str(wrong_total)+' internal-validation scanned pages. Keep it isolated. Prioritise role-confidence calibration, difficult row grouping and crop validation on development data; preserve the current checkpoint and do not start CRNN fine-tuning.') if wrong_total else 'The selected configuration has no wrong acceptances in this internal-validation sample, but remains a development candidate requiring prospective independent validation before deployment.'
 if regression['correct_pages']<baseline['correct_pages'] or (regression['common_cer'] is not None and baseline['common_cer'] is not None and regression['common_cer']>baseline['common_cer']):
  recommendation+=' Clean-image regression also prevents an unconditional replacement of the existing pipeline.'
 write_json(HERE/'reports/final_verification.json',dict(protected_files_unchanged=len(before),locked_source_files=len(lock['locked_files']),model_state_unchanged=completion['model_state_unchanged'],source_protocol_verified=True,all_candidate_cohorts_identical=True,selected_before_validation=winner,tests=tests,training_started=False,github_operations=False))
 long_diagnostic=json.loads((root/'long_sequence_diagnostic.json').read_text());long=long_diagnostic['C4'];positive_clean=index[('validation','positive','clean','C4')]
 text=f'''# NAQSHKASH Stage 2C — robust preprocessing and segmentation

Completed locally on 3 November 2026. All new work is under C:\\NAQSHKASH\\stage2\\preprocessing_improvements. Stage 1, Stage 2A and Stage 2B remain unchanged. No source images/labels were changed or regenerated. No neural model, count classifier or OCR model was trained. No Git operations or cloud services were used.

## Outcome and recommendation

**The predeclared tuning rule selected {winner} before validation was opened. This is an experimental selection, not a deployment endorsement.**

{recommendation}

The experiment deliberately exposes both improvements and wrong acceptances. Higher proposal coverage or lower conditional OCR error alone is not sufficient evidence of a reliable blind page pipeline. No existing application/preprocessing entry point was switched to a candidate.

## Frozen development design

| Pool | Tuning parents | Internal validation parents | Separate clean regression parents |
|---|---:|---:|---:|
| Positive Stage 2A training partition | 182 | 46 | 0 |
| Unexposed general supplementary development | 100 | 50 | 0 |
| Confirmed original Stage 1 training | 0 | 0 | 100 |

There are **478 source parents and 1,612 existing images**: 228 positive parents with one clean and three scanned versions (912 pages), 150 general parents similarly represented (600 pages), and 100 normal clean training pages. Six configurations produce **9,672 page-candidate outcomes**. All variants remain with their source parent. Complete source-image hashes and full logical identities were checked for cross-partition duplication. Shared isolated glyphs are not treated as identical source parents.

All 150 parents exposed in Stage 2B (49 positive, 101 general) were excluded, including the general content-only match. Positive sources come only from the frozen 228-parent training pool; the 49 original positive validation parents and 49 positive test parents were not used for optimization or comparison. The general development pool is drawn from previously unexamined, unmapped parents, and remains **exploratory/development-only**. It is not relabelled as an independent Stage 1 test. Normal clean parents are selected from the actual 150,000-tier Stage 1 training manifest, which contains 79,918 distinct parent IDs.

Data and evaluation thresholds were frozen before experiments. All 478 selected source TXT files agree with JSON tiers and existing canonical Talim rules. Original clean/scanned ZIPs are read in place. The protocol locks source archive hashes, manifests, old split assignments, model, vocabulary and decoder; a software lock preserves the ten files used by the inference implementation. The final audit verified {len(before):,} protected historical project-file hashes. The reproduction launcher additionally verifies the full recorded history before inference. All old Stage 2B reports/manifests are preserved.

Files: `../protocol/design.json`, `../protocol/lock.json`, `../protocol/software_lock.json`, `../protocol/protected_before.json`, `../manifests/parents.csv`, `../manifests/pages.csv`.

## C0–C5 candidates

| Candidate | Change from reference |
|---|---|
| C0 | Exact Stage 2B image-only baseline proposals, original Stage 1 preprocessing |
| C1 | C0 plus image-derived illumination/background normalization |
| C2 | Component-aware tier bands, joint image-only row/role grammar, adaptive whitespace-limited crop padding |
| C3 | Original rotation correction plus guarded baseline-convergence projective correction; original blind grouping |
| C4 | Combined C1/C2/C3 processing |
| C5 | C4 plus gated edge-preserving noise handling and stricter ambiguity rejection |

The robust grouper reconnects short vertical glyph fragments at a scale estimated from connected components. It scores band morphology and the Talim grammar of optional upper, mandatory count, optional lower without an expected number of rows. Count-role evidence uses component height/width and loop structure; it is a hand-designed heuristic, not a trained role classifier. Alternative assignments with close scores can be rejected. Crop padding is clamped to observed whitespace separators rather than allowed to overlap the neighbouring band.

Normalization estimates local paper illumination and divides it out; already-white clean pages bypass this change. Geometry reuses the unchanged Stage 1 rotation estimator and adds a projective attempt only when image baseline support, transform extent and image-only projection-score guards pass. It does not use known augmentation angles/corners. The current projective model is limited to supported baseline convergence; it is not a general page-curl or camera-calibration solution. When both rotational and projective corrections occur, two resamplings are possible, which is a remaining image-quality limitation. C5's denoising is gated by image texture; it does not blindly erase small connected components in the OCR image.

No sample IDs, source recipes, expected counts, label strings, tier annotations or OCR predictions enter `process_image(image, configuration)`. No OCR hypothesis selects or repairs a crop. Ground truth is consulted only by the separate offline evaluator after proposals are fixed.

## How segmentation correctness is measured

The selected source renderer uses 24-pixel margins, eight-pixel line spacing and 20-pixel inter-row spacing. Offline reference tier masks come from those source slots, verified against every clean source pixel and annotated nonempty tier. Only the reference masks are transformed using the existing recorded rotation/perspective recipe; no degraded image dataset is regenerated. These annotations are never available to deployment.

A page is reference-correct only when row count, ordered tier roles and every crop pass: **at least 98% of its intended ideal tier ink retained, and at most 2% neighbouring-tier contamination**. Reference points are mapped forward through the candidate transform, so clipping outside its canvas cannot disappear from the denominator. CTC checks use the original vocabulary and repeat-aware required sequence lengths. OCR is evaluated only for complete pages with correct reference geometry and feasible targets; incomplete/wrong proposals are not artificially scored as character deletions.

- Proposal coverage: algorithm accepts / all input pages.
- Correct segmentation coverage: reference-correct proposals / all input pages.
- Wrong acceptances: algorithm accepts, but offline reference rejects.
- Safe rejections: algorithm abstains; this means no unverified crop is used, **not** proof that a valid page is inherently unreadable.
- Accepted segmentation accuracy: reference-correct / algorithm-accepted.
- Conditional CER and exact tier accuracy: only correctly segmented, CTC-feasible pages.
- Page-exact/all: complete correct segmentation and exact OCR for every tier / all pages, including rejections as unsuccessful.

Recorded geometry is exact at the documented operation level but pixel masks are nearest-neighbour ideal-ink references. Blur halos, JPEG interpolation and subpixel effects are approximations; masks are not manually verified pixel polygons for all scans. The separate component diagnostic and visual panels help expose preservation issues that aggregate scores can miss.

## Candidate selection and internal-validation results

The tuning-only pilot used 20 tuning parents (80 pages); C0 correctly segmented 36/80, robust candidates 60/80, while C5 reduced pilot wrong acceptances from 14 to four. Pilot results are preserved and were not used as validation results.

The complete selection rule was: highest correct scanned tuning-page count, then fewer wrong acceptances, then lower CER on the shared correctly segmented tuning subset. Selection was written after **6,768 tuning page-candidate evaluations** and before any validation page was evaluated. Its saved winner remains {winner}, regardless of subsequent validation findings. This lexicographic rule can favour a small coverage gain despite a large wrong-acceptance increase; the result demonstrates why a precision/safety gate is needed before future promotion. We did not retrospectively relabel another candidate as the winner.

{chr(10).join(cohort_sections)}

### Newly covered long sequences reveal a separate OCR limitation

Do not hide the increased positive conditional CER: C4 scores **27.532% CER** on its expanded scanned positive subset, versus C0's 2.381% on a much smaller subset. The common-subset CER is **3.125% for both**. One newly covered parent, **{long['dense_parent']}**, contributes {long['dense_errors']} of {long['errors']} C4 errors ({100*long['error_share']:.2f}%) and {long['dense_characters']} of {long['characters']} characters ({100*long['character_share']:.2f}%). C4's conditional mean of per-parent CER is {100*long['macro_parent_cer']:.3f}% over {long['scored_parents']} scored parents; this diagnostic does not replace character-weighted CER.

The same parent has **112 deletions on its clean image**, all verified to be missing internal spaces: the non-space sequences match. Across C4's newly covered positive clean-validation subset, CER is **{percentage(positive_clean['cer'])}**, so this is not merely a scanned-background effect. On the long perspective variant, actual count/symbol substitutions and substantial deletions also remain. This exposes limitations of spacing/long-sequence decoding and unresolved distortion even with correct crop structure. C0 avoids the difficult parent by rejecting it; its smaller conditional CER cannot establish superior recognition on that parent.

No spaces were stripped from the targets or headline metrics, and no parent was removed. `long_sequence_diagnostic.json` records the concentration and clean-space-only diagnostic. Next-stage recognition analysis should separate exact spacing from glyph errors while preserving the original evaluation target, and should not automatically interpret this as permission to fine-tune.

On the general validation cohort, C1 isolates a clearer photometric improvement: the same 111/150 correctly segmented pages have CER **6.411% with C0 versus 1.812% with C1**. C4 scores 1.919% on its larger accepted subset and 1.835% on the all-candidate common subset. These comparisons support normalization while distinguishing its OCR benefit from the extra risks introduced by grouping/geometry.

Common-subset rows use the intersection of correctly segmented, OCR-feasible pages across **all six** configurations within each cohort. They compare identical page/tier labels, avoiding the claim that a lower conditional CER caused only by rejecting harder pages is an OCR improvement. Counts and all tuning/clean/validation rows are in `experiments_v1/summary.csv` and JSON. Per-augmentation metrics are in `augmentation_metrics.csv`; no new augmentation categories were invented.

Parent-bootstrap diagnostic intervals for the selected candidate versus C0:

{uncertainty}

These intervals resample source parents (2,000 replicates, seed42), keeping their three correlated variants together. They describe internal development variation, not independent-test or handwriting generalisation.

## Clean-image regression

The table uses 100 confirmed Stage 1 training parents, not the original held-out test. They were not involved in the saved tuning selection. Candidate effects are measured with the same checkpoint on the same clean images.

| Candidate | Correct pages | Wrong acceptances | Rejections | Conditional CER | Exact tiers | Common pages | Common CER |
|---|---:|---:|---:|---:|---:|---:|---:|
{clean_table}

Exact sequence and common-subset results must be read together with page coverage. Also see the positive/general clean-source rows in the machine-readable summary; no full Stage 1 held-out evaluation was rerun or overwritten.

## Failure patterns and limitations

First recorded failure reasons on internal-validation scanned pages:

{chr(10).join(failure_lines)}

Visual review confirms both improvement and regression. For `edge_00242__aug_03_jpeg`, adaptive grouping recovers the upper/count/lower blocks that C0 rejects. For `edge_00270__clean`, a three-dot upper symbol is still split into multiple bands and misassigned. For `edge_00049__aug_03_noise`, apparently reasonable rectangles have incorrect tier roles despite C0 assigning them correctly. Inferred roles are printed on the panels; good-looking boxes alone do not prove correct segmentation.

These reasons are not mutually exclusive underlying causes: an early band failure can hide a later role error. Wrong role/grouping and row assignments can remain despite improved band joining. Guarded perspective correction cannot reliably recover every long, sparse or converging layout. Small confidence differences and similar count/symbol morphology remain difficult. Fine-tuning a CRNN cannot directly repair these image-only grouping mistakes.

The original Stage 2B conditional OCR figures were verified independently from 3,737 saved predictions before this work. Its blind pipeline already failed on 35/49 clean positive parents. Stage 2C's new cohorts and stricter reference-ink checks mean these new results should be compared against **C0 on the same cohort**, not directly against an old Stage 2B headline number.

## Individual dot/component preservation

Before validation began, a separate fixed 90-page diagnostic was registered: ten positive validation parents, ten general validation parents (all clean/scanned versions), and ten normal clean regression parents. It compares C0 with the **already selected** candidate. It cannot select a new winner.

Every ideal transformed ink component of at least three reference pixels is checked individually; less than 95% retention is flagged. Components smaller than three pixels are outside this diagnostic. This is stricter for tiny dots than an aggregate 98% tier-ink threshold.

| Candidate | Pages | Structurally aligned pages | Components checked | Components with loss | Pages preserving all components | Aggregate-pass pages with component loss |
|---|---:|---:|---:|---:|---:|---:|
{component_table}

Component-level counts are conditional on matching structure; rejected/misgrouped pages remain failures of coverage, not automatic preservation successes. Full records and flagged component examples are in `component_audit_pages.csv` and `component_loss_examples.csv`.

## Visual evidence and tests

Before/after panels include successful recovery, wrong acceptance, safe rejection, regression and very wide cases where available. Full-resolution images and left/right details are preserved so tiny dots/strokes can be inspected without relying on a downscaled thumbnail. The selected-candidate gallery is `../examples/selected_candidate/index.md`; earlier C0/C2/C5 experiment panels remain under `experiments_v1/examples/`. Expected labels and evaluated predictions are offline sidecars; no incorrect crop is passed to OCR solely to create an attractive example.

**{tests['passed']} tests passed; {tests['failed']} failed, {tests['errors']} errors, {tests['skipped']} skipped.** This includes new contract/safety/audit tests and relevant unchanged Stage 1 preprocessing, CRNN and CTC regression tests. Evidence is in `tests.json`, `tests.xml` and `tests.txt`. Tests cover annotation-free APIs, source/hash drift rejection, parent isolation, identical candidate cohorts, C0 equivalence, normalization, blank rejection, dot/noise handling, adaptive boundaries, clipped/contaminated crop rejection, reference transformations, source-label offsets, deterministic replay, pre-validation selection, checkpoint/vocabulary/decoder compatibility and historical-file preservation.

Inference used CPU with four PyTorch threads, batch size1, deterministic algorithms, float32 original decoding and no optimizer. Model parameters/buffers were checked equal to the loaded checkpoint after experiments. Independent uncached replay checks passed for all six configurations. The inference loop elapsed {completion['elapsed_seconds']/60:.2f} minutes. Candidate `seconds` values are incremental timings in a shared page processor with exact-pixel prediction caching, **not standalone deployment latency comparisons**.

## Files and next action

All source, manifests, examples, locks and reports are inside this module. No Stage 1/2A/2B files were edited. The original checkpoint SHA256 is `{completion['checkpoint_sha256']}`. Software/source hashes, input checks, tests and outcome counts are recorded in `final_verification.json`.

{recommendation}

The next iteration should add explicit wrong-acceptance limits to a newly registered selection rule and strengthen ambiguous-role rejection using development parents. Illumination normalization should be evaluated as its own low-risk component; combined geometry/grouping changes must not inherit its OCR benefit without meeting segmentation safety gates. Validation has now been observed: do not tune against these reported validation scores and then call them untouched test results. Keep the general cohort exploratory, keep old positive tests protected, and seek a prospective independently defined evaluation before claiming production readiness, camera-scan robustness or real-handwriting performance. No fine-tuning is recommended as the immediate next action.
'''
 (HERE/'reports/stage2c_final_report.md').write_text(text,encoding='utf-8')
 print('Final Stage2C report:',winner,'wrong validation acceptances:',wrong_total,'historical files unchanged:',len(before))

if __name__=='__main__':main()
