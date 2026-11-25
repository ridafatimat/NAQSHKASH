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
