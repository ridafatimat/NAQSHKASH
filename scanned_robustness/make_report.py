"""Summarise measured Stage2B outputs; preserve all protected project files."""
import argparse,json,zipfile
from collections import Counter
from common import *


def pct(value):return 'n/a' if value is None else f'{100*value:.3f}%'
def delta(a,b):return 'n/a' if a is None or b is None else f'{100*(a-b):+.3f} pp'


def main():
 p=argparse.ArgumentParser();p.add_argument('--recommendation',choices=['A','B','C'],required=True);a=p.parse_args()
 root=HERE/'reports/evaluation_v1';result=json.loads((root/'metrics.json').read_text());audit=json.loads((HERE/'reports/dataset_audit.json').read_text())
 before=json.loads((HERE/'reports/protected_before.json').read_text());after=protected_snapshot();assert after==before,'Protected Stage1/Stage2A file changed'
 source_checks={}
 for dataset,config in audit['datasets'].items():
  assert sha256(config['scanned'])==config['archive_sha256'];assert sha256(config['clean'])==config['clean_archive_sha256']
  source_checks[dataset]=dict(scanned_sha256_unchanged=True,clean_sha256_unchanged=True)
 tests=json.loads((HERE/'reports/tests.json').read_text());assert tests['successful']
 coverage=read_csv(root/'page_processing.csv');predictions=read_csv(root/'predictions.csv');failures=read_csv(root/'segmentation_failures.csv')
 assert len(coverage)==audit['selected_pages']*2
 assert len(failures)==sum(r['status']=='failed' for r in coverage)
 examples=read_csv(root/'failure_examples.csv')
 categories=[];tiers=[];summary_lines=[];paired_lines=[];tier_lines=[]
 for cohort,modes in result['groups'].items():
  for mode,e in modes.items():
   m=e['overall'];summary_lines.append(f"| {cohort} | {mode} | {e['pages_processed']}/{e['pages_attempted']} | {m['samples']} | {m['characters']} | {pct(m['cer'])} | {pct(m['exact_accuracy'])} | {m['substitutions']}/{m['deletions']}/{m['insertions']} |")
   clean=e['paired_clean'];scan=e['paired_scanned'];paired_lines.append(f"| {cohort} | {mode} | {scan['samples']} | {pct(clean['cer'])} | {pct(scan['cer'])} | {delta(scan['cer'],clean['cer'])} | {pct(clean['exact_accuracy'])} | {pct(scan['exact_accuracy'])} | {delta(scan['exact_accuracy'],clean['exact_accuracy'])} |")
   for tier,m in e['per_tier'].items():
    tiers.append(dict(cohort=cohort,mode=mode,tier=tier,**m))
    tier_lines.append(f"| {cohort} | {mode} | {tier} | {m['samples']} | {pct(m['cer'])} | {pct(m['exact_accuracy'])} |")
   for dimension in ('augmentation','operation','severity'):
    for value,item in e['per_'+dimension].items():
     categories.append(dict(cohort=cohort,mode=mode,dimension=dimension,value=value,pages_attempted=item['pages_attempted'],pages_processed=item['pages_processed'],**item['metrics']))
 write_csv(root/'category_metrics.csv',categories);write_csv(root/'tier_metrics.csv',tiers)
 baseline_zip=PROJECT/'NAQSHKASH_final_evaluation.zip'
 with zipfile.ZipFile(baseline_zip) as z:
  member=next(n for n in z.namelist() if n.endswith('.json'));frozen=json.loads(z.read(member))['full_test']
 clean_page_rows=read_csv(root/'clean_page_processing.csv')
 clean_lookup={(r['dataset'],r['parent_id'],r['mode']):r for r in clean_page_rows}
 clean_coverage_lines=[];transitions=[]
 for cohort,modes in result['groups'].items():
  for mode,entry in modes.items():
   cp=[r for r in clean_page_rows if r['cohort']==cohort and r['mode']==mode]
   sp=[r for r in coverage if r['cohort']==cohort and r['mode']==mode]
   lost=sum(clean_lookup[(r['dataset'],r['parent_id'],mode)]['status']=='success' and r['status']=='failed' for r in sp)
   prior_failed=sum(clean_lookup[(r['dataset'],r['parent_id'],mode)]['status']=='failed' and r['status']=='failed' for r in sp)
   clean_ok=sum(r['status']=='success' for r in cp)
   clean_coverage_lines.append(f"| {cohort} | {mode} | {clean_ok}/{len(cp)} | {entry['pages_processed']}/{entry['pages_attempted']} | {lost} | {prior_failed} |")
   transitions.append(dict(cohort=cohort,mode=mode,clean_parents_processed=clean_ok,clean_parents_attempted=len(cp),scan_pages_processed=entry['pages_processed'],scan_pages_attempted=entry['pages_attempted'],new_scan_failures_from_clean_success=lost,failed_clean_and_scan=prior_failed))
 write_csv(root/'clean_scan_coverage.csv',transitions)
 worst=[]
 for cohort in ('positive_frozen_test','general_unassigned_exploratory'):
  for mode,e in result['groups'][cohort].items():
   operations=e['per_operation'];seg=sorted(operations.items(),key=lambda pair: (1-pair[1]['pages_processed']/pair[1]['pages_attempted'],pair[1]['pages_attempted']),reverse=True)[:5]
   ocr=sorted([(op,r) for op,r in operations.items() if r['metrics']['characters']>0],key=lambda pair:pair[1]['metrics']['cer'],reverse=True)[:5]
   worst.append(f"### {cohort}, {mode}\n\nLargest page-failure rates (operation groups overlap):\n")
   for op,r in seg:worst.append(f"- {op}: {r['pages_attempted']-r['pages_processed']}/{r['pages_attempted']} failed pages.\n")
   worst.append('\nHighest conditional OCR CER among groups with evaluated characters:\n')
   for op,r in ocr:worst.append(f"- {op}: CER {pct(r['metrics']['cer'])}, exact {pct(r['metrics']['exact_accuracy'])}; {r['metrics']['samples']} tiers / {r['metrics']['characters']} characters.\n")
 identity_lines=[]
 for cohort,count in audit['parents_by_cohort'].items():
  identity_lines.append(f"| {cohort} | {count} | {audit['variants_by_cohort'].get(cohort,0)} | {audit['selected_by_cohort'].get(cohort,0)} |")
 examples_md=['# Representative Stage2B failure examples\n\nFull strings and all errors are in the CSV files. A missing prediction for a segmentation failure means inference was withheld, not an OCR deletion.\n']
 chosen=Counter()
 for row in examples:
  group=(row['cohort'],row['mode'],row['event'])
  if chosen[group]>=2:continue
  chosen[group]+=1
  examples_md.append(f"\n## {row['sample_id']} — {row['mode']} / {row['event']}\n\nCohort: {row['cohort']}; augmentation: {row['augmentation']}. {row['description']}\n\nExpected preview: `{row['truth'][:160]!r}`\n\nPrediction preview: `{row['prediction'][:160]!r}`\n\n![{row['sample_id']}]({Path(row['image_path']).name})\n")
 (root/'examples/index.md').write_text(''.join(examples_md),encoding='utf-8')
 integrity=dict(protected_files_verified=len(before),stage1_and_stage2a_unchanged=True,input_archives=source_checks,
   checkpoint_sha256=result['checkpoint_sha256'],model_state_unchanged=result['model_state_unchanged'],tests=tests,
   deterministic_replays=result['uncached_determinism_replays'],training_started=False)
 (HERE/'reports/final_verification.json').write_text(json.dumps(integrity,indent=2),encoding='utf-8')
 recommendations={
  'A':'A — The existing model is sufficiently robust for the evaluated conditions only. Keep the checkpoint frozen and proceed cautiously; this is not proof for all scans or handwriting.',
  'B':'B — Preprocessing improvements are required before declaring scanned-page readiness. Prioritise robust blind row/tier grouping, role identification, perspective/rotation handling, illumination/background normalization and alignment rejection. Keep the model frozen while testing improvements on a separate development cohort. Reassess residual OCR errors only after crop quality is verified; do not infer that training alone will fix segmentation failures.',
  'C':'C — A future scanned-style fine-tuning experiment is justified by residual recognition errors on verified aligned tiers and their clean-parent comparison. Keep this evaluation frozen, use only training-eligible source parents for learning, and verify scan and clean retention on untouched test cohorts. No training is authorised or run here.'}
 report=f'''# Stage 2B scanned-style robustness evaluation

Measured locally on 3 November 2026 in `C:\\NAQSHKASH\\stage2\\scanned_robustness`. No training, optimizer updates, model/vocabulary modifications, Git operations or cloud execution occurred. All {len(before)} protected project files outside Stage 2B retained their hashes; Stage 1 and Stage 2A are unchanged.

## 1. Archive discovery and integrity

The named general ZIP `talim_scanned_augmented_v1.zip` was not found in the proposed Final dataset folder. The actual matching general dataset is **C:\\FYP Dataset Generation\\datasets\\talim_10000_scanned.zip**, verified by its 30,000-row augmentation manifest and source metadata. The positive archive is **C:\\FYP Dataset Generation\\datasets\\talim_edge_cases_positive_scanned_v1.zip**. No older workspace archive was substituted.

| Dataset | Augmented dataset images | Clean source parents | Variants per parent |
|---|---:|---:|---:|
| General | 30,000 | 10,000 | 3 |
| Positive edge cases | 978 | 326 | 3 |

The newly downloading Final dataset copies were not present at the last check; inference used the already available, fully verified datasets-folder copies above. `download_copy_check.json` records this explicitly, and `check_downloads.py` can compare the downloaded copies against the audited SHA256 values when present. No GitHub access or download was needed for this evaluation.

Both archives also contain one QA montage outside their images directory; montages are not counted as samples. They contain **no embedded TXT labels or per-page label JSON**; references resolve into their clean source ZIPs (`talim_10000.zip` and `talim_edge_cases_positive_v1.zip` in the same datasets folder). Every source inventory hash was checked against those ZIPs. All 30,978 augmented image bytes matched manifest SHA256 hashes, decoded successfully and matched declared dimensions. All clean source labels matched their JSON tiers and existing canonical Talim rules. Clean archive hashes match the recorded official source hashes. No full dataset extraction was needed.

Actual operations: brightness, contrast, gaussian_blur, jpeg, motion_blur, noise, perspective, resolution, rotation, shadow. There are 15 actual single/mixed recipes and mild/medium/strong metadata severities. JPEG output encoding also occurs as a container step for all images; augmentation category jpeg denotes the explicit recipe operation. The recorded coarse readability screen is not an OCR guarantee. No stains/folds/handwriting or other unrecorded category is assumed.

Full counts, recipes, severities, hashes and provenance: `dataset_audit.json`. Every parent and variant, including exclusions: `../manifests/parents.csv` and `all_variants.csv`.

## 2. Source-parent eligibility and leakage

The full original Stage 1 source ZIP was scanned: **100,000 image hashes and 100,000 complete logical-run structures**, against all 10,326 source parents. **No general source image is byte-identical to a Stage 1 source image.** One general parent, all_003613, has exactly the logical structure of Stage 1 test_009409; it lacks byte identity and is reported separately as a test-content-match exploratory case, not as confirmed source provenance.

The other 9,999 general parents have no exact complete-page image or logical-run match in the Stage 1 archive. Their clean-parent mapping is valid within the 10k source dataset, but membership in a previously frozen Stage 1 evaluation split cannot be established. They are **not an independent held-out Stage 1 test**. No random image-level split or new test split was invented.

Positive source identities match the original Stage 2A clean archive and inherit its 326 frozen parent assignments. All 228 source training parents and 49 validation parents (831 variants total) are excluded from held-out inference. All 49 positive test parents, including the previously evaluated/recovered test parents, are retained together with all their 147 variants. This is a robustness follow-up on exposed frozen test parents, not a newly blind benchmark.

| Parent classification | Parents | All variants | Variants evaluated |
|---|---:|---:|---:|
{chr(10).join(identity_lines)}

Before inference, a deterministic **100-parent / 300-variant general exploratory sample** was frozen using lowest SHA256('42:'+parent_id), plus the separate three-variant content-match case. All 147 eligible positive test variants were evaluated. Thus **450 scanned pages** were inferred in two modes; **the entire 30,978-image collection was integrity-audited, not exhaustively inferred**. This distinction limits broad category estimates for the general set. The remaining general images have no declared frozen test eligibility; a full exploratory run or new prospective benchmark requires a separately stated scope.

There are no repeated augmented image hashes, no repeated clean image hashes, and no parent whose variants were classified across splits. Exact source relations and complete logical content were checked; partial-row, near-duplicate and font-level similarity are not ruled out. Manifest: `../manifests/evaluation_pages.csv`; SHA256 `{result['inference_manifest_sha256']}`.

## 3. Frozen model and segmentation modes

Checkpoint: `{CHECKPOINT}`. SHA256 `{result['checkpoint_sha256']}`. Exact original 65-character vocabulary, blank 0, 66 classifier outputs, hidden size 256 and two bidirectional LSTM layers. Model state tensors were checked equal to the loaded checkpoint after inference. CPU, four PyTorch threads, batch size 1, eval/inference mode, no AMP or training, deterministic algorithms enabled. Exact input-crop prediction caching only reuses identical pixels; independent uncached replay checks passed for every cohort/mode with successful predictions.

Common preprocessing is the original Stage 1 pipeline: deskew enabled, cleaning, margin crop and logical row detection; ruling-line removal remains disabled. Original height-64 aspect-preserving bilinear resizing, grayscale /255 normalization, white padding and greedy CTC decoding are unchanged. Clean parent comparisons use the same mode and settings.

**Mode A (blind):** row proposals come from the original image-only detector. Tier bands use the original ink-mask/projection utilities without an expected tier count or label values. One observed band is count; three are upper/count/lower. For two bands, a fixed 1.2 height ratio or 1.15 ink-mass ratio locates the count band; ambiguity is rejected. More than three bands and overlapping padded bands are rejected. These are explicit, untuned pixel heuristics, not a learned count-glyph classifier. No expected rows, expected tier presence, symbol counts, text labels, sample IDs, augmentation parameters or OCR predictions are used to select/retry/rank proposals.

Only **after** blind proposals are fixed does ground truth validate row count and upper/count/lower presence/order. Mismatches reject the page; the proposal is never relabelled or repaired using ground truth. This validation defines the conditional OCR subset and must not be confused with a deployable blind acceptance test.

**Mode B (annotation-assisted):** ground-truth row count may invoke Stage 2A's existing strict-whitespace fallback, and known nonempty tier presence is passed to original tier-band extraction. Raw tier text and leading offsets remain unchanged. The same structural checks, non-overlap and repeat-aware CTC feasibility checks apply. Whole pages are withheld from OCR metrics if any row/tier fails. No ground-truth character values guide image cropping in either mode; only Mode B uses structure.

Alignment here is verified by ordered structural correspondence and crop checks, with representative visual examples available. The augmentation set does not provide pixel-level transformed tier polygons; subtle residual crop/warp contamination may remain even when structure agrees. Recognition differences can therefore include residual image-preprocessing effects, not just classifier errors. No claim of geometrically perfect alignment for every successful tier is made.

## 4. Measured page coverage and conditional OCR

CER is total Levenshtein edits / ground-truth characters, counting spaces. Exact accuracy is complete tier-sequence equality. Edit ties use deterministic substitution/deletion/insertion preference. Failed pages do not become artificial deletion errors and do not enter the OCR denominator. Conditional metric changes must be interpreted with coverage because different modes can accept different pages.

| Cohort | Mode | Pages processed/attempted | Evaluated tiers | Characters | CER | Exact accuracy | S/D/I |
|---|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(summary_lines)}

Rows labelled exploratory must remain separate from the positive frozen-test results. The one-parent content-match sample is too small to support any general claim. Zero evaluated samples would produce n/a metrics, not 0% CER.

### Per-tier OCR results

| Cohort | Mode | Tier | Sequences | CER | Exact accuracy |
|---|---|---|---:|---:|---:|
{chr(10).join(tier_lines)}

Detailed reports: `evaluation_v1/metrics.json`, `metrics.csv`, `tier_metrics.csv`, `category_metrics.csv`, `predictions.csv`, `recognition_errors.csv`, `page_processing.csv`, `segmentation_failures.csv`. Clean parent predictions and clean processing failures are also retained. Segmentation/alignment/CTC failures are separate page records, not recognition-edit counts. Conditional OCR does not equal end-to-end page accuracy.

### Pre-existing blind limitations versus new degradation failures

| Cohort | Mode | Clean parents processed/attempted | Scanned variants processed/attempted | Scan failures whose clean parent passed | Failures on both clean and scan |
|---|---|---:|---:|---:|---:|
{chr(10).join(clean_coverage_lines)}

**Most positive blind-mode failures already occur on clean parents:** the blind extractor handles only 14/49 clean positive parents (28.6%), versus 41/147 scanned variants (27.9%). Of its 106 scanned failures, 105 also fail on the corresponding clean parent; only one is a new failure on an otherwise accepted clean parent. Thus poor blind positive coverage cannot be attributed entirely to scan corruption or to the CRNN. The prior excellent clean OCR result was annotation-assisted and conditional on segmentation.

In the general exploratory sample, blind clean coverage is 92/100 parents and scan coverage 224/300 variants; 52 newly failing variants came from clean parents that passed, while 24 fail in both conditions. Assisted clean coverage is complete for these selected cohorts, exposing genuine additional scan-processing losses (56 general and five positive variants). Coverage transitions are in `evaluation_v1/clean_scan_coverage.csv`.

## 5. Comparison with clean Stage 1

The preserved full clean synthetic Stage 1 test report records **{frozen['samples']:,} tiers**, CER **{pct(frozen['cer'])}**, exact accuracy **{pct(frozen['exact_accuracy'])}**. That frozen report is context, not an identical sample distribution. Stage 2A's 97 unique positive clean test tiers scored 0% CER/100% exact, but Stage 2B includes all raw parent-tier instances and augmentation repetitions, so its denominators differ.

For a fairer comparison, the same exact clean source parents were run through the same preprocessing/mode/model. The table below pairs only scanned tiers with an aligned clean counterpart. Clean predictions are repeated per matching scanned tier instance so character/sequence denominators are identical within each pair; it does not count those repetitions as independent parents. Failed clean pages are reported and excluded from pairing.

| Cohort | Mode | Paired tiers | Clean CER | Scan CER | CER change | Clean exact | Scan exact | Exact change |
|---|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(paired_lines)}

Do not interpret differences against the full historical clean baseline as solely caused by degradations; use paired results and coverage together. Neither source-parent label uncertainty nor segmentation failures can be cured merely by substituting a headline OCR number.

## 6. Actual degradation groups and examples

Operation groups overlap for mixed recipes, so rankings show associations in this dataset, not isolated causal effects. Many groups contain correlated variants from the same parent and modest denominators. Severity values are generator metadata, not a calibrated perceptual scale. Complete recipe, operation and severity metrics include coverage and character denominators in the JSON/CSV.

{chr(10).join(worst)}

### Specific reviewed findings

Actual combined recipes matter: in the general exploratory assisted cohort, **brightness+noise** has 19.277% CER (19/19 pages structurally processed), whereas **noise alone** has 0.105% CER. Do not describe the mixed-recipe result as the isolated effect of noise or brightness. The positive assisted subset has 14.118% CER for perspective+shadow (9/11 pages processed) and 11.429% for rotation (9/11). Small groups, accepted-page selection and correlated variants limit these comparisons.

A visually reviewed, structurally aligned general upper-tier example expects `BH`: its clean prediction is `BH`, while the brightness+noise crop predicts `U`. The two symbol clusters remain visible. The full-crop median/p95 intensities are 255/255 in the clean reference and 193/199 in the scanned crop, consistent with a visibly grey grainy background. This suggests testing background/illumination normalization on development data; these full-crop statistics are not a background-only measurement or a causal ablation. The matched clean crop and `qualitative_review.json` are saved alongside the examples. A reviewed positive three-dot `A` crop is predicted as `U`, showing that recognition errors also remain even when the symbol is visibly present.

Edit attribution confirms that general assisted errors include 107 deleted spaces out of 246 total deletions, plus 120 substitutions and one insertion. These spaces are part of the fixed OCR target and were not stripped to improve results. The positive assisted set has 14 substitutions and 11 deletions, no insertions. Attribution is deterministic and checked against every saved per-sample edit count (`edit_attribution.json`). Residual recognizer weakness and preprocessing/background sensitivity both need to be distinguished from rejected-page segmentation failures.

Representative failures were saved without modifying originals. `evaluation_v1/failure_examples.csv` includes expected text, predictions where inference ran, source parent, recipe, mode and descriptions; `evaluation_v1/examples/index.md` previews selected images. Segmentation examples have no prediction because OCR was withheld. Recognition examples contain the exact evaluated tier crop, expected label and prediction. All errors, not only selected examples, are retained in their machine-readable CSVs.

## 7. Tests and preservation

**{tests['tests_run']} automated tests passed**, zero failures/errors/skips. Tests cover archive counts and source-image/label/JSON links; parent-level frozen eligibility and source-identity conservatism; deterministic exploratory selection; blind interface/ground-truth separation; rejection instead of relabelling; label offsets/CTC feasibility; CER/exact/no-data calculations; strict checkpoint/vocabulary compatibility; complete failure accounting; and uncached deterministic inference replay/model immutability. Test evidence: `tests.json` and `tests.txt`.

Every selected variant has exactly one page outcome per mode, and each successful page's tier count matches prediction rows. All dataset images were checked before inference. {len(before)} protected project-file hashes and all audited scanned/clean archive hashes were rechecked after inference; Stage 1/Stage 2A files and source archives remain unchanged. No model state tensor changed. `final_verification.json` records these assertions.

## 8. Recommendation

**{recommendations[a.recommendation]}**

Regardless of the next development choice, preserve this manifest, checkpoint and all results. Future model/preprocessing choices must use development-eligible parents and a new prospective benchmark where appropriate; this test has now been observed. Obtain documented frozen Stage 1 source correspondence or establish a separately named benchmark before calling the general set an independent test. Synthetic scan-style transformations do not demonstrate real camera-scan coverage or handwriting recognition.
'''
 (HERE/'reports/stage2b_final_report.md').write_text(report,encoding='utf-8')
 print('Report written; protected files verified:',len(before))
 print('\n'.join(summary_lines))

if __name__=='__main__':main()
