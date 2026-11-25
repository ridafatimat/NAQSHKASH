"""Final all-writer fit, then one W05 evaluation and unchanged regression gates."""
import argparse,json,time
from pathlib import Path
import v5b_engine as e
def main():
 p=argparse.ArgumentParser();p.add_argument('--resume',nargs='?',const='latest');p.add_argument('--fold-run');p.add_argument('--smoke-test',action='store_true');args=p.parse_args();cfg=e.config();base=e.ROOT/'local_run'/('smoke_tests' if args.smoke_test else 'outputs');base.mkdir(exist_ok=True)
 if args.resume:
  runs=sorted(base.glob('final_*'));run=runs[-1] if args.resume=='latest' and runs else Path(args.resume).resolve();assert run.is_dir() and run.parent.resolve()==base.resolve()
  meta=json.loads((run/'run_set.json').read_text());assert meta['config']==cfg,'Config changed; start a new final run';foldrun=Path(meta['fold_run'])
 else:
  choices=[p.parent for p in sorted(base.glob('folds_*/completed.json'))];foldrun=Path(args.fold_run).resolve() if args.fold_run else choices[-1] if choices else None
  if foldrun is None:raise RuntimeError('Run run_folds.py first, or specify --fold-run')
  assert foldrun.resolve().parent==base.resolve()
  run=base/('final_'+str(time.time_ns()));run.mkdir();e.dump(run/'run_set.json',dict(kind='final',smoke=args.smoke_test,fold_run=str(foldrun),config=cfg,W05_accessed=False))
 selection=json.loads((foldrun/'fold_selection.json').read_text());assert selection['smoke']==args.smoke_test
 groups=cfg['folds']['configurations'][selection['configuration']];updates=20 if args.smoke_test else selection['updates']
 result=e.train_run(run/'training',cfg,'final',groups,max_updates=updates,smoke=args.smoke_test,resume=bool(args.resume));m,v,_=e.v2.load_model(result['best_checkpoint']);originals,replay,dev=e.inputs()
 if args.smoke_test:
  # Exercise both 14/13-tier report paths without opening W05 during a smoke test.
  va=[r for r in originals if r['sample_type']=='original' and r['writer_id']=='W01'][:14];excluded_id=va[-1]['id'];dev=[r for cohort in sorted({r['cohort'] for r in dev}) for role in ['upper','count','lower'] for r in [x for x in dev if x['cohort']==cohort and x['role']==role][:2]]+[r for r in dev if r['cohort']=='positive_development' and len(r['truth'])>=65][:2]
 else:va=e.read(e.V5/'W05_evaluation_inputs.csv');excluded_id='all_003801_row_00_count';assert len(va)==14 and {r['writer_id'] for r in va}=={'W05'}
 once=run/'evaluation_once.json'
 if not once.exists():
  marker=run/'evaluation_started.json'
  if marker.exists():assert json.loads(marker.read_text())['checkpoint_sha256']==e.sha(result['best_checkpoint'])
  else:e.dump(marker,dict(checkpoint_sha256=e.sha(result['best_checkpoint']),selection_sha256=e.sha(foldrun/'fold_selection.json'),smoke=args.smoke_test))
  metrics,raw,masked=e.evaluate(m,v,va,run/'evaluation','candidate_once',True);subset={d:e.all_metrics([r for r in rr if r['id']!=excluded_id]) for d,rr in [('raw',raw),('role_mask',masked)]};e.dump(once,dict(full_14=metrics,subset_13=subset,smoke=args.smoke_test,W05_used=not args.smoke_test));e.write(run/'evaluation/count_errors.csv',e.count_errors(raw,'raw')+e.count_errors(masked,'role_mask'))
  e.dump(run/'evaluation/near_exact.json',{d:{str(n):[r['id'] for r in rr if r['edit_distance']==n] for n in [1,2]} for d,rr in [('raw',raw),('role_mask',masked)]})
 else:metrics=json.loads(once.read_text())['full_14']
 _,sr,sm=e.evaluate(m,v,dev,run/'regression','candidate',True);reg={'raw':e.synthetic_metrics(sr,dev),'role_mask':e.synthetic_metrics(sm,dev)}
 frozen,_,_=e.v2.load_model();_,br,bm=e.evaluate(frozen,v,dev,run/'regression','frozen_E1',True);basereg={'raw':e.synthetic_metrics(br,dev),'role_mask':e.synthetic_metrics(bm,dev)}
 gates={d:{'original':e.synthetic_gates(reg[d],basereg[d],cfg['gates']),'no_whitespace_view':e.synthetic_gates(reg[d],basereg[d],cfg['gates'],True)} for d in ['raw','role_mask']}
 if args.smoke_test:basehand,_,_=e.evaluate(frozen,v,va,run/'evaluation','frozen_surrogate')
 else:
  # Reuse already verified frozen raw W05 baseline; never use W05 to choose settings.
  old=json.loads((e.legacy.V/'final_comparison.json').read_text());basehand={'raw':old['approaches']['A']['validation']}
 base=basehand['raw'];candidate=metrics[cfg['final']['promotion_decoder']]
 handwritten=dict(minimum_three_fewer_handwritten_edits=candidate['overall']['errors']<=base['overall']['errors']-cfg['gates']['minimum_error_reduction'],no_whitespace_CER_not_worse=candidate['overall']['cer_no_whitespace']<=base['overall']['cer_no_whitespace'],exact_accuracy_not_lower=candidate['overall']['exact_accuracy']>=base['overall']['exact_accuracy'],both_validation_sources_not_worse=all(candidate['per_source'][s]['cer']<=base['per_source'][s]['cer'] for s in base['per_source']))
 decoder=cfg['final']['promotion_decoder'];passed=all(handwritten.values()) and gates[decoder]['original']['passed'] and (not cfg['final']['require_no_whitespace_synthetic_gates'] or gates[decoder]['no_whitespace_view']['passed'])
 decision=dict(passed=passed and not args.smoke_test,smoke=args.smoke_test,handwritten_checks=handwritten,synthetic_gate_views=gates,operational_default='frozen_E1' if not passed or args.smoke_test else 'candidate_eligible_no_operational_file_changed',checkpoint=result['best_checkpoint'],checkpoint_sha256=e.sha(result['best_checkpoint']),deployment_performed=False)
 e.dump(run/'synthetic_regression_results.json',dict(candidate=reg,frozen=basereg,gates=gates));e.dump(run/'decision.json',decision);e.dump(run/'completed.json',dict(completed=True,smoke=args.smoke_test,W05_used=not args.smoke_test));print('FINAL COMPLETE',run,json.dumps(decision),flush=True)
 e.write(run/'synthetic_regression_results.csv',[dict(model=model,decoder=decoder,cohort=cohort,**values) for model,data in [('frozen_E1',basereg),('candidate',reg)] for decoder,cohorts in data.items() for cohort,values in cohorts.items()])
if __name__=='__main__':main()
