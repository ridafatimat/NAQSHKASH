"""Measured development CTC loss curves, separate from CER curves."""
import os,sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from context import E
os.environ['MPLCONFIGDIR']=str(E/'reports/matplotlib_cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(2,1,figsize=(9,7))
baseline=json.loads((E/'metrics/baseline_selection_summary.json').read_text())
for ax,cohort in zip(axes,['positive_development','clean_validation']):
 for name in ['e1','e2']:
  history=json.loads((E/'metrics'/f'{name}_training_history.json').read_text())
  points=[r for r in history if 'development' in r]
  ax.plot([r['step'] for r in points],[r['development'][cohort]['ctc_loss'] for r in points],marker='o',label=name.upper())
 ax.axhline(baseline[cohort]['ctc_loss'],linestyle='--',color='gray',label='E0')
 ax.set_title(cohort);ax.set_xlabel('Optimizer updates');ax.set_ylabel('Mean CTC loss');ax.grid(alpha=.2);ax.legend()
fig.suptitle('Actual loss on frozen exposed selection-development inputs')
fig.tight_layout();fig.savefig(E/'reports/development_loss_curves.png',dpi=160);fig.savefig(E/'reports/development_loss_curves.svg')
print('Development CTC loss curves saved')

