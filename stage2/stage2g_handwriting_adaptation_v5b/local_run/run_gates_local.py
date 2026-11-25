"""Inference-only gates for the user-selected update-750 checkpoint. No training."""
import argparse,json,time
from pathlib import Path
import v5b_engine as e
ROOT=e.ROOT
def report(selected,gate_results=None,decision=None,w05=None):
 gen=ROOT/'generated_only';base=json.loads((gen/'baseline/training_105_metrics.json').read_text())
 lines=['# NAQSHKASH V5b — selected update 750','', 'Training was stopped at the user’s request. Update 750 is the selected generated-only checkpoint; later checkpoints are retained but are not selected. No full overnight fold/final training has been run.','',f"Selected checkpoint: `{selected['checkpoint']}`",f"SHA256: `{selected['checkpoint_sha256']}`",'', '| Model/update | Raw overall CER | Raw macro-writer CER | Raw no-space CER | Raw exact /105 | Mask macro-writer CER | Mask exact /105 |','|---|---:|---:|---:|---:|---:|---:|']
 entries=[('frozen_E1',base)]
 for file in sorted((gen/'training').glob('attempt_*/evaluations/step_*_metrics.json')):entries.append((file.stem.replace('_metrics',''),json.loads(file.read_text())))
 for name,m in entries:lines.append(f"| {name} | {m['raw']['overall']['cer']:.4%} | {m['raw']['macro_writer']['cer']:.4%} | {m['raw']['overall']['cer_no_whitespace']:.4%} | {m['raw']['overall']['exact_tiers']} | {m['role_mask']['macro_writer']['cer']:.4%} | {m['role_mask']['overall']['exact_tiers']} |")
 selected_metrics=json.loads((gen/'training/attempt_001/evaluations/step_00750_metrics.json').read_text())
 lines+=['', '| Selected decoder | Writer | CER | No-space CER | Exact tiers |','|---|---|---:|---:|---:|']
 for decoder,m in selected_metrics.items():
  for writer,r in m['per_writer'].items():lines.append(f"| {decoder} | {writer} | {r['cer']:.4%} | {r['cer_no_whitespace']:.4%} | {r['exact_tiers']}/{r['tiers']} |")
 lines+=['','Per-role, per-writer, count-error and raw/masked prediction files for every evaluated checkpoint remain in `generated_only/training/attempt_001/evaluations/`. All 105 originals were used for development evaluation, never for generated-only gradient updates. W05 is not used for selection.','', '## Synthetic gates and decision','']
 if gate_results is None:lines+=['**PENDING — full gates and W05 have not been run.** The current safe initialization is frozen E1. Run `local_run/run_gates_local.py` locally before the folds. A dry run does not produce a scientific PASS/FAIL or evaluate W05.']
 else:
  lines += [f"**{'PASS' if decision['passed'] else 'FAIL'}**. The starting checkpoint is {'update 750' if decision['gates_passed'] else 'frozen E1'}.",'',f"The 105-tier raw macro-writer CER improvement is {decision['macro_writer_CER_improvement']*100:.2f} percentage points. The original >=3-point handwriting condition is {decision['macro_improvement_passed']}. All synthetic gate views passed: {decision['gates_passed']}.",'','PASS requires >=3 percentage points lower raw macro-writer CER than frozen E1, plus every synthetic CER/exact gate across both decoders and both whitespace views. Initialization follows the synthetic-gate result, as requested, independently of informational W05 results. Existing numerical gate tolerances are unchanged.','', '| Decoder | Cohort | E1 CER | Selected CER | E1 no-space CER | Selected no-space CER |','|---|---|---:|---:|---:|---:|']
  for decoder in ['raw','role_mask']:
   for cohort,r in gate_results['candidate'][decoder].items():
    b=gate_results['frozen'][decoder][cohort];lines.append(f"| {decoder} | {cohort} | {b['cer']:.4%} | {r['cer']:.4%} | {b['cer_no_whitespace']:.4%} | {r['cer_no_whitespace']:.4%} |")
  lines+=['', 'Individual original/no-whitespace checks are saved in `local_run/gate_outputs/update_00750/synthetic_gates.json`.','', '## Informational W05','', '| Tiers | Decoder | CER | No-space CER | Exact |','|---|---|---:|---:|---:|']
  for view,data in w05.items():
   for decoder,m in data.items():
    r=m['overall'];lines.append(f"| {view} | {decoder} | {r['cer']:.4%} | {r['cer_no_whitespace']:.4%} | {r['exact_tiers']}/{r['tiers']} |")
  lines+=['','W05 was evaluated only after the gate decision and initialization were fixed. Both decoders use the same posterior for each image. The 13-tier view excludes only `all_003801_row_00_count`. Cached posteriors prevent repeated forwards on resume.']
 lines+=['', '## QA and package validation','', 'Safe4 selected QA round 1: core darkness median 42.5 versus real 39.72; detached-mark diameter median 5.0 px, with accepted equivalent diameters guarded to 4–7 px. Visibility rejections were 4.42% of all proposals and 13.89% of rendered proposals. The <5% acceptance used all proposals; the stricter rendered-only screen failed. All three rounds and both denominators are retained in `quality_safe4/`.', '', 'Earlier local fold/final scripts passed their bounded 20-update smoke tests, with no W05 access; those temporary outputs were deleted. The new gates script is inference-only and has its own tiny synthetic-only dry-run verification. No model was deployed; frozen E1 remains the operational default.', '', 'Global safe4 style calibration used all four training writers. A Step 2 initialization was also selected using all 105 development tiers; subsequent folds exclude held-out real gradient updates but are not fully nested validation. Count-feature error categories and compact-mark measurements are heuristic rather than independent reader verification.']
 (ROOT/'final_report_v5b.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--dry-run',action='store_true');parser.add_argument('--resume',action='store_true');args=parser.parse_args();cfg=e.config();e.c.seed(cfg['seed'])
 selected=json.loads((ROOT/'generated_only/selected_checkpoint.json').read_text());assert selected['selected_step']==750 and e.sha(selected['checkpoint'])==selected['checkpoint_sha256']
 if args.dry_run:
  folder=ROOT/'local_run/_gates_dry_run';assert not folder.exists();folder.mkdir()
  if not (ROOT/'final_report_v5b.md').exists():report(selected)
  config_hash=e.sha(ROOT/'local_run/config.json');report_hash=e.sha(ROOT/'final_report_v5b.md')
 else:
  folder=ROOT/'local_run/gate_outputs/update_00750'
  if folder.exists() and not args.resume:raise RuntimeError('Outputs already exist; use --resume to reuse cached inference and W05 results')
  folder.mkdir(parents=True,exist_ok=True)
  manifest=folder/'selected_checkpoint.json'
  if manifest.exists():assert json.loads(manifest.read_text())['checkpoint_sha256']==selected['checkpoint_sha256']
  else:e.dump(manifest,selected)
 _,_,dev=e.inputs()
 if args.dry_run:
  dev=[next(r for r in dev if r['cohort']=='clean_historical_regression'),next(r for r in dev if r['cohort']=='clean_validation'),next(r for r in dev if r['cohort']=='positive_development' and len(r['truth'])<65),next(r for r in dev if r['cohort']=='positive_development' and len(r['truth'])>=65)]
  assert len({r['id'] for r in dev})==4
 frozen,v,_=e.v2.load_model();frozen.requires_grad_(False);candidate,_,_=e.v2.load_model(selected['checkpoint']);candidate.requires_grad_(False)
 print('Inference only:',len(dev),'synthetic tiers per model; no training',flush=True)
 _,br,bm=e.evaluate(frozen,v,dev,folder/'synthetic','frozen_E1',True);_,cr,cm=e.evaluate(candidate,v,dev,folder/'synthetic','selected_750',True)
 baseline={'raw':e.synthetic_metrics(br,dev),'role_mask':e.synthetic_metrics(bm,dev)};reg={'raw':e.synthetic_metrics(cr,dev),'role_mask':e.synthetic_metrics(cm,dev)}
 checks={d:{'original':e.synthetic_gates(reg[d],baseline[d],cfg['gates']),'no_whitespace':e.synthetic_gates(reg[d],baseline[d],cfg['gates'],True)} for d in ['raw','role_mask']};gates_passed=all(r['passed'] for views in checks.values() for r in views.values())
 if args.dry_run:
  assert e.sha(ROOT/'local_run/config.json')==config_hash and e.sha(ROOT/'final_report_v5b.md')==report_hash
  e.dump(ROOT/'local_run/run_gates_local_dry_run_verification.json',dict(passed=True,synthetic_tiers_per_model=4,synthetic_ids=[r['id'] for r in dev],model_forwards=8,W05_evaluated=False,training_updates=0,gate_views_exercised=['raw','role_mask','whitespace','no_whitespace'],config_unchanged=True,full_report_unchanged=True,test_outputs_deleted=False,selected_checkpoint_sha256=selected['checkpoint_sha256']))
  print('DRY RUN PASSED: four synthetic tiers/model; no W05; no training; config unchanged');return
 e.dump(folder/'synthetic_gates.json',dict(candidate=reg,frozen=baseline,checks=checks,all_passed=gates_passed))
 base105=json.loads((ROOT/'generated_only/baseline/training_105_metrics.json').read_text());m105=json.loads((ROOT/'generated_only/training/attempt_001/evaluations/step_00750_metrics.json').read_text());improvement=base105['raw']['macro_writer']['cer']-m105['raw']['macro_writer']['cer'];macro_ok=improvement>=cfg['generated']['minimum_macro_CER_improvement']-1e-12
 decision=dict(passed=gates_passed and macro_ok,gates_passed=gates_passed,macro_improvement_passed=macro_ok,macro_writer_CER_improvement=improvement,checkpoint=selected['checkpoint'],checkpoint_sha256=selected['checkpoint_sha256'],selected_step=750,decision_unix=time.time(),W05_used_for_selection=False,initialization_rule='update 750 iff all synthetic gates pass; otherwise frozen E1')
 before=folder/'decision_before_W05.json'
 if before.exists():
  old=json.loads(before.read_text());assert old['checkpoint_sha256']==decision['checkpoint_sha256'] and old['gates_passed']==gates_passed;decision=old
 else:e.dump(before,decision)
 # Initialization is decided before any W05 access, using only the synthetic gates.
 init_decision=dict(decision,passed=gates_passed);e.dump(ROOT/'generated_only/decision.json',init_decision)
 cfg['initialization']='step2' if gates_passed else 'frozen_e1';cfg['gates_local']['status']='complete';cfg['gates_local']['gates_passed']=gates_passed;cfg['gates_local']['starting_checkpoint']=selected['checkpoint'] if gates_passed else str(e.c.E/'checkpoints/selected_model/best.pt');e.dump(ROOT/'local_run/config.json',cfg)
 once=folder/'W05_once.json'
 if once.exists():w05=json.loads(once.read_text())
 else:
  va=e.read(e.V5/'W05_evaluation_inputs.csv');assert len(va)==14 and {r['writer_id'] for r in va}=={'W05'}
  e.dump(folder/'W05_started.json',dict(checkpoint_sha256=selected['checkpoint_sha256'],decision_sha256=e.sha(before)))
  metrics,raw,masked=e.evaluate(candidate,v,va,folder/'W05','selected_750_once',True);w05={'14_tiers':metrics,'13_tiers':{d:e.all_metrics([r for r in rows if r['id']!='all_003801_row_00_count']) for d,rows in [('raw',raw),('role_mask',masked)]}};e.dump(once,w05)
 report(selected,dict(candidate=reg,frozen=baseline,checks=checks),decision,w05);e.dump(folder/'completed.json',dict(completed=True,decision=decision,W05_informational_only=True));e.dump(ROOT/'generated_only/status.json',dict(phase='stopped_training_gates_complete',selected_step=750,gates_pending_local_execution=False,W05_evaluated=True,training_restarted=False));print('COMPLETE:', 'PASS' if decision['passed'] else 'FAIL','; initialization:',cfg['initialization'])
if __name__=='__main__':main()
