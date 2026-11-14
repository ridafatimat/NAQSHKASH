"""Sequential V5 queue; does not touch historical experiment outputs."""
import sys,time,json,traceback
from pathlib import Path
import stage2g_v5 as s
s.c.seed(42)
def status(phase,**extra):
 s.put('run_status.json',dict(phase=phase,updated_unix=time.time(),**extra))
 print('V5 QUEUE',phase,flush=True)
try:
 status('waiting_for_corrected_augmentation_QA')
 while not (s.O/'quality_safe3/augmentation_quality.json').exists():time.sleep(10)
 for conf in s.CONFIGS:
  tag='pilot_safe3_'+conf['name']
  if not (s.O/'training_logs'/(tag+'_completion.json')).exists():
   status('capacity_pilot',configuration=conf['name'])
   s.train_run(conf,tag,heldout='W04',max_steps=400,min_steps=400)
 status('full_pilot_regression_assessment');s.pilot_assessment()
 status('generated_only_2000_updates');s.generated_only()
 status('four_writer_folds_three_configurations');s.folds()
 status('writer_macro_selection_and_review_ablation');s.fold_selection()
 status('final_W05_and_regression_evaluation');s.final_evaluation()
 status('complete',report=str(s.O/'final_comparison_report.md'))
except BaseException as exc:
 status('failed',error=repr(exc),traceback=traceback.format_exc())
 raise
