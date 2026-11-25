"""Two-writer, two-configuration overnight folds. Only --smoke-test is run by Codex."""
import argparse,json,time
from pathlib import Path
import v5b_engine as e
def select(run,cfg,results,smoke=False):
 table=[];summary=[]
 for name in cfg['folds']['configurations']:
  scores=[];steps=[]
  for writer in cfg['folds']['heldout_writers']:
   r=results[name+'__'+writer];best=next(x for x in r['evaluation_history'] if x['step']==r['best_step']);m=best['metrics'][cfg['folds']['selection_decoder']]['overall'];scores.append(m);steps.append(r['best_step']);table.append(dict(configuration=name,heldout_writer=writer,best_step=r['best_step'],trained_updates=r['steps'],checkpoint=r['best_checkpoint'],**m))
  summary.append(dict(configuration=name,macro_writer_CER=sum(x['cer'] for x in scores)/len(scores),macro_writer_CER_no_whitespace=sum(x['cer_no_whitespace'] for x in scores)/len(scores),macro_writer_exact_accuracy=sum(x['exact_accuracy'] for x in scores)/len(scores),best_updates=steps))
 chosen=min(summary,key=lambda x:(x['macro_writer_CER'],x['macro_writer_CER_no_whitespace'],-x['macro_writer_exact_accuracy']));duration=20 if smoke else max(cfg['folds']['min_updates'],int(round(float(e.np.median(chosen['best_updates']))/cfg['checkpoint_every'])*cfg['checkpoint_every']))
 selection=dict(configuration=chosen['configuration'],updates=duration,fold_summary=summary,rule='unweighted mean CER across W01 and W06; then no-whitespace CER, then exact accuracy; no W05 data',final_duration_rule=cfg['final']['duration'],smoke=smoke,W05_used=False,created_unix=time.time())
 e.write(run/'writer_independent_results.csv',table);e.dump(run/'fold_selection.json',selection);return selection
def main():
 p=argparse.ArgumentParser();p.add_argument('--resume',nargs='?',const='latest');p.add_argument('--smoke-test',action='store_true');args=p.parse_args();cfg=e.config()
 base=e.ROOT/'local_run'/('smoke_tests' if args.smoke_test else 'outputs');base.mkdir(exist_ok=True)
 if args.resume:
  choices=sorted(base.glob('folds_*'))
  run=choices[-1] if args.resume=='latest' and choices else Path(args.resume).resolve()
  assert run.is_dir() and run.parent.resolve()==base.resolve(),'Resume folder must be a fold run under '+str(base)
 else:run=base/('folds_'+str(time.time_ns()));run.mkdir()
 if args.resume:assert json.loads((run/'run_set.json').read_text())['config']==cfg,'Config changed; start a new fold set'
 results={};e.dump(run/'run_set.json',dict(kind='folds',smoke=args.smoke_test,config=cfg,W05_accessed=False))
 for name,groups in cfg['folds']['configurations'].items():
  for writer in cfg['folds']['heldout_writers']:
   key=name+'__'+writer;print('FOLD',key,'smoke',args.smoke_test,flush=True)
   results[key]=e.train_run(run/key,cfg,'fold',groups,heldout=writer,max_updates=20 if args.smoke_test else cfg['folds']['max_updates'],smoke=args.smoke_test,resume=bool(args.resume))
 selection=select(run,cfg,results,args.smoke_test);e.dump(run/'completed.json',dict(completed=True,smoke=args.smoke_test,selection=selection));print('FOLDS COMPLETE',run,json.dumps(selection),flush=True)
if __name__=='__main__':main()
