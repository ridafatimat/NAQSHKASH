import sys,time,json
from pathlib import Path
import v5b_engine as e
def flatten(label,metrics):
 out=[]
 for decoder,m in metrics.items():
  out.append(dict(checkpoint=label,decoder=decoder,group='overall',name='all',**m['overall']))
  for group in ['per_role','per_writer']:
   out.extend(dict(checkpoint=label,decoder=decoder,group=group,name=name,**row) for name,row in m[group].items())
 return out
def run():
 cfg=e.config();out=e.ROOT/'generated_only';out.mkdir(exist_ok=True);assert json.loads((e.ROOT/'quality_safe4/acceptance.json').read_text())['passed']
 rows,replay,dev=e.inputs();originals=[r for r in rows if r['sample_type']=='original'];e.dump(out/'config.json',cfg)
 e.dump(out/'status.json',dict(phase='frozen_baseline',W05_used_for_selection=False))
 baseline_path=out/'baseline/training_105_metrics.json'
 if not baseline_path.exists() or not (out/'baseline/synthetic_cohorts.json').exists():
  m,v,_=e.v2.load_model()
  if not baseline_path.exists():
   base,raw,masked=e.evaluate(m,v,originals,out/'baseline','training_105');e.write(out/'baseline/count_errors.csv',e.count_errors(raw,'frozen_E1_raw')+e.count_errors(masked,'frozen_E1_role_mask'))
  if not (out/'baseline/synthetic_cohorts.json').exists():
   _,sr,sm=e.evaluate(m,v,dev,out/'baseline','synthetic');e.dump(out/'baseline/synthetic_cohorts.json',{'raw':e.synthetic_metrics(sr,dev),'role_mask':e.synthetic_metrics(sm,dev)})
 base=json.loads(baseline_path.read_text());basereg=json.loads((out/'baseline/synthetic_cohorts.json').read_text())
 e.dump(out/'status.json',dict(phase='generated_only_training',W05_used_for_selection=False))
 result=e.train_run(out/'training',cfg,'generated',['cnn','rnn','classifier'],resume='--resume' in sys.argv)
 metricsrows=flatten('frozen_E1',base)
 for evaluation in result['evaluation_history']:metricsrows.extend(flatten('step_'+str(evaluation['step']),evaluation['metrics']))
 e.write(out/'training_writer_results.csv',metricsrows)
 e.dump(out/'status.json',dict(phase='selected_checkpoint_regression',selected_checkpoint=result['best_checkpoint'],W05_used_for_selection=False))
 m,v,_=e.v2.load_model(result['best_checkpoint']);_,rr,rm=e.evaluate(m,v,dev,out/'selected','synthetic',True);reg={'raw':e.synthetic_metrics(rr,dev),'role_mask':e.synthetic_metrics(rm,dev)}
 best=next(r for r in result['evaluation_history'] if r['step']==result['best_step']);improvement=base['raw']['macro_writer']['cer']-best['metrics']['raw']['macro_writer']['cer'];clean=e.synthetic_gates(reg['raw'],basereg['raw'],cfg['gates'],clean_only=True)
 decision=dict(passed=improvement>=cfg['generated']['minimum_macro_CER_improvement']-1e-12 and clean['passed'],rule='PASS iff raw macro-writer CER on the 105 originals improves by >=0.03 versus raw frozen E1 AND the existing clean synthetic CER/exact gates hold; W05 is informational only',macro_writer_CER_improvement=improvement,frozen_macro_writer_CER=base['raw']['macro_writer']['cer'],selected_macro_writer_CER=best['metrics']['raw']['macro_writer']['cer'],clean_gate=clean,checkpoint=result['best_checkpoint'],checkpoint_sha256=e.sha(result['best_checkpoint']),selected_step=result['best_step'],decision_unix=time.time(),W05_used=False,clipping_rate=result['clipping_rate'],seconds_per_update=result['seconds_per_update'])
 if (out/'decision.json').exists():
  previous=json.loads((out/'decision.json').read_text());assert previous['checkpoint_sha256']==decision['checkpoint_sha256'] and previous['passed']==decision['passed'];decision=previous
 else:e.dump(out/'decision.json',decision)
 gate_views={decoder:{'original':e.synthetic_gates(reg[decoder],basereg[decoder],cfg['gates']),'no_whitespace_view':e.synthetic_gates(reg[decoder],basereg[decoder],cfg['gates'],True)} for decoder in ['raw','role_mask']};e.dump(out/'synthetic_regression_results.json',dict(candidate=reg,frozen=basereg,gates=gate_views))
 flat=[]
 for model,data in [('frozen_E1',basereg),('selected',reg)]:
  for decoder,cohorts in data.items():flat.extend(dict(model=model,decoder=decoder,cohort=cohort,**r) for cohort,r in cohorts.items())
 e.write(out/'synthetic_regression_results.csv',flat)
 # This is the only generated-only candidate W05 inference; both decoders share each forward pass.
 if not (out/'W05_once.json').exists():
  va=e.read(e.V5/'W05_evaluation_inputs.csv');assert len(va)==14 and {r['writer_id'] for r in va}=={'W05'}
  marker=out/'W05_once_started.json'
  if marker.exists():assert json.loads(marker.read_text())['checkpoint_sha256']==decision['checkpoint_sha256']
  else:e.dump(marker,dict(checkpoint_sha256=decision['checkpoint_sha256'],decision_sha256=e.sha(out/'decision.json'),timestamp_unix=time.time()))
  wm,wr,ww=e.evaluate(m,v,va,out/'informational_W05','candidate_once',True);sub={decoder:e.all_metrics([r for r in rows if r['id']!='all_003801_row_00_count']) for decoder,rows in [('raw',wr),('role_mask',ww)]};e.dump(out/'W05_once.json',{'14_tiers':wm,'13_tiers':sub,'informational_only':True,'selection_decision_already_fixed_sha256':e.sha(out/'decision.json')})
 e.dump(out/'status.json',dict(phase='complete',passed=decision['passed'],decision_file=str(out/'decision.json')))
 print('GENERATED ONLY COMPLETE',json.dumps(decision),flush=True)
if __name__=='__main__':e.c.seed(42);run()
