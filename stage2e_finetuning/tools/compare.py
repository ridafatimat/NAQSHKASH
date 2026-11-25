"""Final paired comparison, predeclared model selection and error evidence."""
import sys,json,math,shutil,html,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from context import *
from metrics import aggregate
from selection import eligible,rank
from collections import defaultdict,Counter
def grouped(rows,keys):
 groups=defaultdict(list)
 for r in rows:groups[tuple(r[k] for k in keys)].append(r)
 return [dict(zip(keys,k),**aggregate(g)) for k,g in sorted(groups.items())]
def main():
 summaries={name:json.loads((E/'metrics'/f'{name}_summary.json').read_text()) for name in ['baseline','e1','e2']}
 samples={name:read(E/'metrics'/f'{name}_per_sample.csv') for name in summaries}
 ids=[r['id'] for r in samples['baseline']]
 assert all([r['id'] for r in ps]==ids for ps in samples.values())
 names={'baseline':'E0','e1':'E1','e2':'E2'}
 for name in ['e1','e2']:
  result=json.loads((E/'reports'/f'{name}_completion.json').read_text());assert result['original_unchanged'] and result['steps']>0
 candidates=[name for name in ['e1','e2'] if eligible(summaries[name],summaries['baseline'])]
 selected=min(candidates,key=lambda n:rank(summaries[n])) if candidates else 'baseline'
 source=CHECKPOINT if selected=='baseline' else Path(json.loads((E/'reports'/f'{selected}_completion.json').read_text())['candidate_checkpoint'])
 folder=E/'checkpoints/selected_model'
 if folder.exists():raise RuntimeError('Preserve frozen selection; do not overwrite')
 folder.mkdir();target=folder/'best.pt';shutil.copy2(source,target);assert digest(source)==digest(target)
 m,v,state=model_load(target);assert all(torch.isfinite(t).all() for t in m.state_dict().values())
 selection=dict(selected=names[selected],source_checkpoint=str(source),selected_checkpoint=str(target),sha256=digest(target),original_sha256=EXPECTED,selected_is_fine_tuned=selected!='baseline',selection_rule=json.loads((E/'configs/training_design.json').read_text())['config']['selection'],eligibility={names[n]:eligible(summaries[n],summaries['baseline']) for n in ['e1','e2']},full_development_used=True,independent_test=False,untouched_test_evaluations=0)
 dump('reports/selected_checkpoint.json',selection);dump('checkpoints/selected_model/lineage.json',selection)
 comparisons=[]
 for name,summary in summaries.items():
  for cohort,value in summary.items():
   if cohort=='runtime':continue
   comparisons.append(dict(candidate=names[name],cohort=cohort,**{k:value[k] for k in ['tiers','parents','characters','cer','exact_accuracy','macro_parent_cer','space_normalized_diagnostic_cer','substitutions','deletions','insertions','ctc_loss']}))
 write('metrics/experiment_comparison.csv',comparisons)
 allrows=[dict(r,candidate=names[n]) for n,rows in samples.items() for r in rows]
 write('metrics/by_length.csv',grouped(allrows,['candidate','cohort','length_bucket']))
 write('metrics/by_distortion.csv',grouped(allrows,['candidate','cohort','augmentation']))
 write('metrics/by_role.csv',grouped(allrows,['candidate','cohort','role']))
 write('metrics/by_parent.csv',grouped(allrows,['candidate','cohort','parent_id']))
 # Page end-to-end accounting includes all D1 rejects/wrong proposals.
 ledger=read(E/'manifests/page_outcomes.csv');ends=[]
 for name,rows in samples.items():
  bycase=defaultdict(list)
  for r in rows:bycase[(r['cohort'],r['case_id'])].append(r)
  for cohort in sorted({p['cohort'] for p in ledger}):
   pages=[p for p in ledger if p['cohort']==cohort];correct=[p for p in pages if p['correct']=='True']
   exact=sum(bool(bycase[(cohort,p['case_id'])]) and all(r['exact']=='True' for r in bycase[(cohort,p['case_id'])]) for p in correct)
   ends.append(dict(candidate=names[name],cohort=cohort,pages=len(pages),parents=len({p['parent_id'] for p in pages}),correct_segmentation=len(correct),wrong_segmentation=sum(p['proposal']=='True' and p['correct']!='True' for p in pages),segmentation_rejected=sum(p['proposal']!='True' for p in pages),end_to_end_exact_pages=exact,end_to_end_exact_accuracy=exact/len(pages)))
 write('metrics/end_to_end_pages.csv',ends)
 # Long spacing analysis retains exact original labels.
 long=[]
 for name,rows in samples.items():
  for r in rows:
   if r['cohort']=='positive_development' and r['parent_id']=='edge_00323':
    long.append(dict(candidate=names[name],id=r['id'],case_id=r['case_id'],role=r['role'],truth=r['truth'],prediction=r['prediction'],truth_spaces=r['truth'].count(' '),predicted_spaces=r['prediction'].count(' '),net_missing_spaces=r['truth'].count(' ')-r['prediction'].count(' '),nonspace_sequences_equal=r['truth'].replace(' ','')==r['prediction'].replace(' ',''),substitutions=r['substitutions'],deletions=r['deletions'],insertions=r['insertions']))
 write('metrics/long_sequence_predictions.csv',long)
 # Standalone scientific curves from actual logged updates.
 os.environ['MPLCONFIGDIR']=str(E/'reports/matplotlib_cache')
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(3,1,figsize=(9,11))
 for name in ['e1','e2']:
  history=json.loads((E/'metrics'/f'{name}_training_history.json').read_text())
  axes[0].plot([h['step'] for h in history],[max(h['train_loss'],1e-10) for h in history],label=names[name],alpha=.7)
  evaluated=[h for h in history if 'development' in h]
  for ax,cohort in [(axes[1],'positive_development'),(axes[2],'clean_validation')]:
   ax.plot([h['step'] for h in evaluated],[100*h['development'][cohort]['cer'] for h in evaluated],marker='o',label=names[name])
 axes[0].set_yscale('log');axes[0].set_ylabel('Training CTC loss (log)')
 baseline_selection=json.loads((E/'metrics/baseline_selection_summary.json').read_text())
 for ax,cohort in [(axes[1],'positive_development'),(axes[2],'clean_validation')]:
  ax.axhline(100*baseline_selection[cohort]['cer'],color='gray',linestyle='--',label='E0');ax.set_ylabel(cohort+' CER (%)')
 for ax in axes:ax.set_xlabel('Optimizer updates');ax.grid(alpha=.2);ax.legend()
 fig.suptitle('Measured fine-tuning; exposed selection-development cohort');fig.tight_layout();fig.savefig(E/'reports/training_curves.png',dpi=160);fig.savefig(E/'reports/training_curves.svg');plt.close(fig)
 # Error gallery: gains, regressions, remaining errors and the long parent.
 before={r['id']:r for r in samples['baseline']};after={r['id']:r for r in samples[selected]};manifest={r['id']:r for r in read(E/'manifests/evaluation.csv')}
 errors=lambda r:sum(int(r[k]) for k in ['substitutions','deletions','insertions'])
 keys=[k for k in ids if before[k]['cohort']=='positive_development']
 choices=[]
 for category,ordered in [('gains',sorted([k for k in keys if errors(before[k])>errors(after[k])],key=lambda k:errors(before[k])-errors(after[k]),reverse=True)[:5]),('regressions',sorted([k for k in keys if errors(after[k])>errors(before[k])],key=lambda k:errors(after[k])-errors(before[k]),reverse=True)[:5]),('remaining',sorted(keys,key=lambda k:errors(after[k]),reverse=True)[:5]),('long',[k for k in keys if before[k]['parent_id']=='edge_00323'])]:
  for k in ordered:
   if k not in [c[1] for c in choices]:choices.append((category,k))
 sections=[]
 page_metadata={(p['dataset'],p['case_id']):p for p in dc.pages()}
 for i,(category,k) in enumerate(choices):
  r=manifest[k];im=cv2.imread(r['path'],0);imagepath=E/'galleries'/f'error_{i:03}.png';cv2.imwrite(str(imagepath),im)
  detail=imagepath.with_name(imagepath.stem+'_detail.png');cv2.imwrite(str(detail),im[:,:min(1000,im.shape[1])])
  b=before[k];a=after[k]
  page=page_metadata[('positive',r['case_id'])]
  with zipfile.ZipFile(dc.old.SOURCE_CONFIG['positive'][page['kind']]) as archive:raw=archive.read(page['image_member'])
  assert hashlib.sha256(raw).hexdigest()==page['image_sha256']
  original=cv2.imdecode(np.frombuffer(raw,np.uint8),0);originalpath=E/'galleries'/f'original_error_{i:03}.png';cv2.imwrite(str(originalpath),original)
  originaldetail=originalpath.with_name(originalpath.stem+'_detail.png');cv2.imwrite(str(originaldetail),original[:,:min(1000,original.shape[1])])
  sections.append(f'<section><h2>{html.escape(category)} — {html.escape(k)}</h2><p>Original page</p><a href="{originalpath.name}"><img src="{originaldetail.name}" style="max-width:95%;max-height:350px"></a><p>Frozen processed tier</p><a href="{imagepath.name}"><img src="{detail.name}" style="max-width:95%"></a><pre>Expected: {html.escape(r["truth"])}\nE0: {html.escape(b["prediction"])}\n{names[selected]}: {html.escape(a["prediction"])}</pre><p>Errors E0={errors(b)}, selected={errors(a)}; exact E0={b["exact"]}, selected={a["exact"]}; parent={r["parent_id"]}</p></section>')
 (E/'galleries/error_comparison.html').write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px sans-serif;margin:24px}pre{white-space:pre-wrap;overflow-wrap:anywhere}section{border-top:1px solid #aaa;padding:10px}</style><h1>Paired recognition errors: identical frozen tier inputs</h1><p>All labels and meaningful spaces are preserved. This is exposed development evidence, not an independent test.</p>'+''.join(sections),encoding='utf-8')
 def pct(x):return f'{100*x:.3f}%'
 lines=['# Stage2E experiment comparison','', '| Cohort | Candidate | Tiers | Parents | CER | Exact tiers | Macro parent CER | S / D / I |','|---|---|---:|---:|---:|---:|---:|---|']
 for r in comparisons:
  lines.append(f"| {r['cohort']} | {r['candidate']} | {r['tiers']} | {r['parents']} | {pct(r['cer'])} | {pct(r['exact_accuracy'])} | {pct(r['macro_parent_cer'])} | {r['substitutions']} / {r['deletions']} / {r['insertions']} |")
 lines+=['','All candidates use identical fixed tier inputs, vocabulary and greedy decoder. This is a paired common-subset comparison. Seven failed pages in the 484-page D1 ledger remain visible separately; no valid hard evaluation tier is discarded. Added clean-validation tiers are an original Stage1 validation subset, not complete page evaluation. Historical/general/positive development was previously exposed; no independent test is claimed.','',f"Selected: **{names[selected]}**, checkpoint {target}. Eligibility: {json.dumps(selection['eligibility'])}. Original SHA256 stays {EXPECTED}.", '', 'Fine-tuning is bounded by the frozen optimizer budget and early stopping, not a claim of convergence or an exhaustive hyperparameter search. E1/E2 initialize independently from E0. No E3 was run. Inference and model-selection metrics preserve spaces. Space-normalized metrics are diagnostics and never select the model.']
 (E/'reports/experiment_comparison.md').write_text('\n'.join(lines),encoding='utf-8')
 (E/'reports/model_selection.md').write_text('# Development-only checkpoint selection\n\n'+json.dumps(selection,indent=2)+'\n\nPer-run checkpoints are selected on the fixed 180-tier exposed selection-development cohort. Completed E1/E2 candidates are compared on the full fixed cohorts using the same predeclared clean/parent/long-sequence gates. The original baseline is retained if no candidate qualifies. No untouched test was evaluated. Preflight checkpoints are never eligible.\n',encoding='utf-8')
 (E/'reports/error_analysis.md').write_text('# Remaining recognition and semantic errors\n\nSelected '+names[selected]+'. Exact per-tier long predictions/spaces are in metrics/long_sequence_predictions.csv. All substitutions, deletions and insertions, parent aggregates, length buckets and actual distortion groups are preserved in metrics/.\n\nThe source canonical_row implementation in talim/talim_rules.py uses horizontal character positions and ljust/rjust padding to align symbols with encoded repetition counts; width=max(len(symbol),len(encoded_count)). Missing internal spaces can therefore shift symbol/count association. They are not assumed harmless. No production whitespace normalization or carpet-generation change was made, and no unverified rendered-carpet equivalence is claimed. Any downstream alignment decoder must be separately validated.\n\nThe long perspective edge_00323 variant remains rejected by frozen D1 and is not credited with OCR improvement. Other correctly segmented long variants remain in every recognition comparison. Segmentation coverage is identical for E0/E1/E2; fine-tuning cannot repair D1 row/tier mistakes.\n\nSee galleries/error_comparison.html and reports/training_curves.png for paired predictions and measured curves.\n',encoding='utf-8')
 dump('reports/comparison_complete.json',dict(selected=names[selected],paired_tiers=len(ids),identical_input_order=True,selected_checkpoint=str(target),checkpoint_sha256=digest(target),independent_test=False))
 print('FINAL SELECTION',json.dumps(selection),flush=True)
if __name__=='__main__':main()

