"""Finalize measured reports and test evidence; no candidate modifications."""
from dcore import *
from collections import defaultdict,Counter
import xml.etree.ElementTree as ET
def main():
 tree=ET.parse(ROOT/'reports/tests.xml').getroot();suites=list(tree) if tree.tag=='testsuites' else [tree]
 totals={k:sum(int(s.get(k,0)) for s in suites) for k in ['tests','failures','errors','skipped']}
 totals.update(passed=totals['tests']-totals['failures']-totals['errors']-totals['skipped'],deselected=1,seconds=sum(float(s.get('time',0)) for s in suites),initial_focused_failure_corrected=True)
 dump('reports/tests.json',totals)
 (ROOT/'reports/test_results.md').write_text(f"""# Stage 2D test results

Final comprehensive run: **{totals['passed']} passed, {totals['failures']} failures, {totals['errors']} errors, {totals['skipped']} skipped, 1 explicitly deselected**. JUnit elapsed {totals['seconds']:.2f} seconds.

The 29 Stage 2D tests cover exact role-search alternatives, ambiguity rejection, unusual valid layouts, rows, extent-limited grouping, dots and source-pixel preservation, crop boundaries, tiny components in the wrong intended tier, CTC/spacing diagnostics, deterministic frozen inference, parent isolation, source-label and crop hashes, fine-tuning eligibility, historical preservation and reproduction namespaces. Relevant unchanged Stage 2B/2C and Stage 1 preprocessing, model-forward and CTC numerics tests are included. No training loop test was run.

One old Stage 2C test, test_software_and_protected_project_are_unchanged, assumes equality of the entire project tree and therefore rejects any authorized new stage directory. Its source was not changed; it is explicitly deselected. Stage 2D verifies inclusion and exact hashes of all 7,051 protected historical files, all 486 inventoried Stage 2C artifacts and the frozen Stage 2D runtime files. This replaces the obsolete whole-tree equality assertion without relaxing protection of any old file.

Initial focused run: 11/12 passed; disconnected-dot edge_00001 failed because a 5px gap exceeded a 4px limit. The training-parent gap correction is recorded, with original code/results/configuration and failure evidence preserved in reports/attempt1. The focused rerun passed 12/12 before final selection.

Evidence: test_command.ps1, focused_tests.txt, tests.txt, tests.xml and tests.json. Temporary files stayed under this module, bytecode/cache writes were disabled, and the local Stage 2A pytest dependency directory was read only. No historical test source, model, labels, datasets or Git state was changed.
""",encoding='utf-8')
 summary=json.loads((ROOT/'reports/candidate_summary.json').read_text());readiness=json.loads((ROOT/'manifests/readiness_summary.json').read_text())
 records=read(ROOT/'reports/evaluation_v1/pages.csv');preds=read(ROOT/'reports/evaluation_v1/predictions.csv')
 oldrecords=[r for r in read(C/'reports/experiments_v1/pages.csv') if r['candidate'] in ['C4','C5']]
 oldpreds=[r for r in read(C/'reports/experiments_v1/predictions.csv') if r['candidate'] in ['C4','C5']]
 from summarize_d1 import metric
 groups=defaultdict(list)
 for r in oldpreds+preds:groups[tuple(r[k] for k in ['candidate','partition','dataset','kind','role'])].append(r)
 roles=[dict(zip(['candidate','partition','dataset','kind','role'],k),**metric(ps),**{f:sum(int(r[f]) for r in ps) for f in ['substitutions','deletions','insertions']}) for k,ps in sorted(groups.items())]
 csvout('reports/role_ocr_metrics.csv',roles)
 augs=[]
 for k in sorted({tuple(r[f] for f in ['candidate','partition','dataset','augmentation']) for r in records+oldrecords if r['kind']=='scanned'}):
  matches=lambda r:tuple(r[f] for f in ['candidate','partition','dataset','augmentation'])==k and r['kind']=='scanned'
  rs=[r for r in records+oldrecords if matches(r)];ps=[r for r in preds+oldpreds if matches(r)]
  augs.append(dict(zip(['candidate','partition','dataset','augmentation'],k),pages=len(rs),parents=len({r['parent_id'] for r in rs}),correct=sum(r['correct_segmentation']=='True' for r in rs),wrong=sum(r['proposal']=='True' and r['correct_segmentation']!='True' for r in rs),rejected=sum(r['proposal']!='True' for r in rs),**metric(ps)))
 csvout('reports/augmentation_metrics.csv',augs)
 failures=[r for r in records if r['correct_segmentation']!='True'];csvout('reports/d1_failures.csv',failures)
 cats=[]
 for k in sorted({(r['partition'],r['dataset'],r['kind'],r['reason']) for r in failures}):
  rs=[r for r in failures if (r['partition'],r['dataset'],r['kind'],r['reason'])==k]
  cats.append(dict(zip(['partition','dataset','kind','reason'],k),images=len(rs),parents=len({r['parent_id'] for r in rs}),example=rs[0]['case_id']))
 csvout('reports/d1_failure_categories.csv',cats)
 comparison=ROOT/'reports/candidate_comparison.md'
 text=comparison.read_text(encoding='utf-8').split('\n\n## Important regressions and subset effects')[0]
 text+="""

## Important regressions and subset effects

D1's scanned tuning result is 830/846 correct, with three wrong acceptances, all from edge_00310. C5 was 598/846 correct with 75 wrong acceptances. The frozen safety rule passed before validation.

On the already exposed scanned validation comparison, D1 is 137/138 positive and 149/150 general correct, with zero wrong acceptances and one rejection in each dataset. Every one of the 46 positive and 50 general parents has at least one correctly segmented variant; 45/46 positive and 49/50 general parents have all three scanned variants correct. These are correlated development variants, not 288 independent source parents.

Do not treat zero scanned validation wrong acceptances as universal safety: across all 1,612 pages, D1 has **1,585 correct acceptances, five wrong acceptances and 22 rejections**. Four wrong pages belong to tuning parent edge_00310 (one clean and all three scans); one wrong page is exposed-validation clean edge_00307. These are valid lower-symbol layouts. Visual review shows a disconnected lower symbol split into multiple count rows. No post-validation fix or reselection was made.

Compared with C5, D1 newly rejects five previously correct general tuning scans from two parents (all_005744, all_009950), plus the clean all_009950 page. It makes the clean and two scanned edge_00310 pages worse; the third scan was already wrong under C5. These losses remain in d1_change_counts.csv/regression_analysis.json.

The one lost positive scanned validation page is **edge_00323__aug_02_perspective**, rejected for unassigned foreground. C4/C5 scored 479 errors on its 758 characters. Therefore the reduction from ~27.53% to **12.371% positive conditional CER** is substantially a difficult-page rejection/subset effect, not a recognition improvement. On the shared C4/C5/D1 subset of 92 positive pages (1,866 characters), all three have **13.076% CER**. The retained long clean page still has 112 missing internal spaces.

Expanded general conditional CER increases from **1.855% C5 to 2.546% D1**. D1 retains every one of C5's 134 correctly segmented general scans, adding 15 harder pages. The 134-page common subset is unchanged at **1.855% CER**; the additional 671 characters contribute 54 errors. Recognition on the shared subset did not worsen, but newly covered distorted images remain difficult for the frozen CRNN.

Normal clean regression improves from 96 to 98 correct pages, with wrong acceptances reduced from three to zero relative to C5. No previously correct normal clean page is lost. Two previously wrong pages are corrected; another wrong page is rejected. Common-subset clean CER is unchanged at 0.177% on 96 shared pages. One additional general tuning clean page is rejected, and positive tuning edge_00310 regresses; clean regression is therefore not a universal no-regression claim.

## Continuity with Stage 2C's historical common subset

The historical six-candidate intersection is retained completely by D1: 32 positive scanned validation pages (96 characters), 111 general scanned pages (4,305 characters), and 88 normal clean pages (3,667 characters). On these identical pages: positive CER remains 3.125% for C0/C4/C5/D1; general CER is 6.411% C0, 1.835% C4 and 1.765% C5/D1; clean CER remains 0.191%. See historical_common_subset.csv. These denominators differ from the broader three-candidate common subset above.

## Component and structural verification

The identical 90-page diagnostic has 70 structurally aligned pages for C4, 70 for C5 and 87 for D1. No geometric component losses were found among aligned cases. The approximate processed-ink content diagnostic finds four weak components for each historical candidate and zero for D1; this is not a manual pixel-shape guarantee.

An additional intended-tier audit over all 1,585 structurally aligned D1 pages checks **65,881 components >=3px**, finding zero components below 95% retention in their intended crop. This is stronger than a union-of-crops test: a dot in the wrong tier cannot be certified merely because some crop contains it. A dedicated adversarial test demonstrates that aggregate 98% ink rules alone can miss that error.

Raw row-count hypothesis accuracy, accepted row coverage/accuracy, exact tier-role structure, proposal coverage, wrong acceptances, valid rejections, page-exact/all, role-specific OCR, actual augmentation recipes and per-parent results are saved separately. Every denominator is explicit. Component masks derive from the frozen synthetic renderer and recorded transformations and approximate blur/interpolation; reference components <3px are not certified.

## Fine-tuning recommendation

The verified dataset subset contains **714 images from 179 training parents and 1,794 tier crops** (1,655 unique crop hashes; 145 distinct target strings). Fourteen images from five training parents need manual review; these are not eligible. The 184 exposed internal-validation images from 46 parents remain excluded from training. Ready and review parent counts overlap because variants can have different outcomes. All 228 positive source-parent transcriptions were independently rechecked against TXT, JSON and canonical Talim rules. Human-reviewed count is zero.

Proceed next to a controlled loader/resource validation and a separately approved fine-tuning pilot using only the verified subset, with parent isolation, a clean-training replay strategy, a frozen validation/stopping protocol and explicit spacing/long-sequence metrics. The current data are small targeted edge cases, not proof that fine-tuning will help. Keep the 14 review pages excluded until resolved; do not automatically train them from D1 acceptance.

D1 is **not deployment-approved**: residual valid-layout mistakes, exposed validation, synthetic-only degradation and long OCR spacing/glyph errors remain. Obtain prospective independent evaluation before deployment/generalization claims. No model training, checkpoint changes, original dataset changes, Git operations, negative-example training or handwriting training occurred.
"""
 comparison.write_text(text,encoding='utf-8')
 # Keep the completed pre-implementation diagnosis immutable; extra evidence is a separate file.
 (ROOT/'reports/failure_analysis_addendum.md').write_text("""# Post-freeze diagnostic addendum

The pre-implementation failure_analysis.md remains byte-identical to its diagnosis-complete hash.

Residual failures are concentrated in two valid lower-symbol parents: edge_00310 (tuning, four variants) and edge_00307 (exposed-validation, clean only). Full-resolution gallery inspection confirms splitting of disconnected lower glyph parts into extra count rows. This is a component-grouping/row-structure failure; no OCR is scored on those wrong crops. It does not establish a dynamic-programming search defect.

The long perspective rejection edge_00323 leaves unassigned foreground under the proposal projection threshold. It is a valid positive page and remains a coverage failure, not a successful rejection of an invalid document. It previously contributed 479/758 character errors; its exclusion explains much of the conditional CER reduction. No labels/spaces were rewritten, and no candidate was tuned on this observed outcome.

Scale/merge signatures in the original diagnosis overlap; they are evidence, not disjoint causal counts. Geometry/background artifacts are not blamed without an isolating experiment. Remaining long-sequence mistakes on aligned crops are frozen recognizer/spacing/distortion findings; they were not repaired by choosing text-compatible crops.

The 90-page comparison and intended-tier component audit strengthen preservation evidence. Their ideal-ink and >=3px scope does not certify every subpixel/damaged mark or every real-world capture. The five wrong acceptances across all cohorts prevent automatic deployment approval despite zero wrong scanned validation proposals.
""",encoding='utf-8')
 dump('reports/final_summary.json',dict(date_local='2026-11-06',pages=1612,parents=478,D1_correct=1585,D1_wrong=5,D1_rejected=22,positive_scanned_validation=dict(pages=138,parents=46,correct=137,wrong=0,rejected=1),general_scanned_validation=dict(pages=150,parents=50,correct=149,wrong=0,rejected=1),clean_regression=dict(pages=100,parents=100,correct=98,wrong=0,rejected=2),readiness=readiness,tests=totals,independent_test_claim=False,deployment_approved=False,training_performed=False,git_operations=False))
 print('Reports finalized:',totals,flush=True)
if __name__=='__main__':main()

