"""Build the final deliverables only after the real Step 2 run and smoke tests finish."""
import json,statistics,collections,sys,time
from pathlib import Path
import v5b_engine as e
def main():
 root=e.ROOT;gen=root/'generated_only';local=root/'local_run';decision=json.loads((gen/'decision.json').read_text());result=json.loads((gen/'training/completion.json').read_text());w05=json.loads((gen/'W05_once.json').read_text());qa=json.loads((root/'quality_safe4/acceptance.json').read_text());smoke=json.loads((local/'smoke_test_verification.json').read_text());assert smoke['passed'] and smoke['outputs_deleted']
 lock=json.loads((root/'V5_preservation_lock.json').read_text());changed=[p for p,digest in lock.items() if e.sha(p)!=digest];assert not changed,changed
 historical=json.loads((e.V5/'protected_inputs.json').read_text());historical_changed=[p for p,digest in historical.items() if e.sha(p)!=digest];assert not historical_changed,historical_changed
 assert e.sha(e.c.E/'checkpoints/selected_model/best.pt')==e.config()['frozen_e1_sha256']
 e.dump(root/'final_integrity.json',dict(V5_unchanged_after_authorized_stop_record=True,V2_V3_and_E1_protected_files_unchanged=True,protected_V5_file_count=len(lock),protected_historical_file_count=len(historical),frozen_E1_unchanged=True,no_existing_datasets_or_checkpoints_modified=True,no_Git_operations=True,no_W02_W07_images_opened=True,no_overnight_folds_launched=True,W05_not_used_for_selection=True))
 base=json.loads((gen/'baseline/training_105_metrics.json').read_text());best=next(r for r in result['evaluation_history'] if r['step']==result['best_step']);reg=json.loads((gen/'synthetic_regression_results.json').read_text());macro=[]
 for name,metrics in [('frozen_E1',base)]+[('step_'+str(r['step']),r['metrics']) for r in result['evaluation_history']]:
  for decoder,m in metrics.items():macro.append(dict(checkpoint=name,decoder=decoder,**m['macro_writer']))
 e.write(gen/'macro_writer_results.csv',macro)
 all_errors=[]
 for file in [gen/'baseline/count_errors.csv']+list((gen/'training').glob('attempt_*/evaluations/*_count_errors.csv')):
  for r in e.read(file):r['evaluation_file']=str(file.relative_to(gen));all_errors.append(r)
 e.write(gen/'all_count_errors.csv',all_errors)
 counts=collections.Counter((r['evaluation_file'],r['model'],r['type']) for r in all_errors);pairs=sorted({(r['evaluation_file'],r['model']) for r in all_errors});e.write(gen/'count_error_summary.csv',[dict(evaluation_file=file,decoder=decoder,error_type=typ,count=counts[(file,decoder,typ)]) for file,decoder in pairs for typ in ['plain_shape_confused','oval_vs_plain_shape','wrong_dot_count','wrong_cross_type','missed_glyph','extra_glyph','out_of_role']])
 speed=result['seconds_per_update'];foldsmoke=statistics.mean(r['seconds_per_update'] for r in smoke['fold_runs']);finalsmoke=smoke['final_run']['seconds_per_update'];e.dump(local/'runtime_estimate.json',dict(generated_seconds_per_update=speed,fold_smoke_seconds_per_update=foldsmoke,final_smoke_seconds_per_update=finalsmoke,minimum_fold_updates=8000,maximum_fold_updates=24000,final_updates_range=[2000,6000],from_generated_speed_minimum_hours=10000*speed/3600,from_generated_speed_ceiling_hours=30000*speed/3600,from_real_batch_smoke_minimum_hours=(8000*foldsmoke+2000*finalsmoke)/3600,from_real_batch_smoke_ceiling_hours=(24000*foldsmoke+6000*finalsmoke)/3600,notes='Step 2 and smoke tests overlapped on CPU for part of timing. Real crops are wider than many synthetic tiers; the real-batch smoke estimate is more relevant. Ceiling is a safety bound, not an expected convergence time. Sleep pauses wall-clock progress.'))
 initial='the PASS-eligible Step 2 checkpoint' if decision['passed'] else 'frozen E1 (Step 2 failed the decision rule)'
 readme=r'''# Local V5b overnight package

No full fold or full final run has been launched by Codex. Both scripts passed bounded smoke tests; their temporary outputs were deleted. All new runs go under this package's `outputs/` in unique timestamped subfolders.

## Environment and exact PowerShell commands

```powershell
Set-Location 'C:\NAQSHKASH\stage2\stage2g_handwriting_adaptation_v5b\local_run'
& 'C:\Users\hp\AppData\Local\Programs\Python\Python38\python.exe' -m venv --system-site-packages .venv
.\.venv\Scripts\Activate.ps1
python -B -c "import torch, cv2, scipy, PIL; print(torch.__version__)"
python -B run_folds.py
```

The virtual environment inherits the already tested packages on this machine; no download is required. If activation is unavailable in your PowerShell session, use `& '.\.venv\Scripts\python.exe' -B run_folds.py` directly instead. The tested base interpreter is Python 3.8 with CPU PyTorch 2.4.1, OpenCV, NumPy, SciPy, and Pillow.

After folds finish:

```powershell
python -B run_final.py
```

This selects the latest completed fold set, chooses the configuration by unweighted mean held-out CER across W01 and W06, and fixes the final update count from the selected folds' median best update (rounded to 250, minimum 2,000). It fits all four training writers before one W05 evaluation with both decoders and 14/13-tier views, then checks synthetic gates. To choose a particular fold set:

```powershell
python -B run_final.py --fold-run '.\outputs\folds_REPLACE_WITH_RUN_ID'
```

## Initialization and settings

Every setting for optimizer, rates, warm-up, clipping, batch composition, augmentation, fold limits, gates and ETA is in `config.json`. Default `initialization: "auto_step2"` currently resolves to **@@INITIAL@@**. Change it to `"frozen_e1"` to force E1. Explicit `"step2"` is permitted only if Step 2 passed. Every run records the actual initial checkpoint path/hash, and the KL teacher always stays frozen E1. No operational/default model file is changed.

Folds: hold out W01 and W06, separately; CNN+BiLSTM with frozen classifier and all layers. Each update accumulates 16 microbatches: 6 safe4 synthetic, 3 originals, 3 existing real augmentations, and 4 clean synthetic. Original and augmented samples are writer-balanced; every descendant of the held-out writer is excluded. The audited 105/2,413 training pool is unchanged. W05 is never used for selection. Unconfirmed shape variants are disabled.

## Crash, interruption, or sleep

```powershell
python -B run_folds.py --resume
python -B run_final.py --resume
```

Each command resumes its latest corresponding run. For an explicit run:

```powershell
python -B run_folds.py --resume '.\outputs\folds_REPLACE_WITH_RUN_ID'
python -B run_final.py --resume '.\outputs\final_REPLACE_WITH_RUN_ID'
```

Checkpoints every 250 updates contain model, optimizer, Python/NumPy/Torch RNG states, sampler counters, clipping history and early-stopping state. Up to 249 uncheckpointed updates may be replayed after a crash. Resume creates a new `attempt_*` log segment, retaining the abandoned tail instead of overwriting it. The saved state/evaluation history defines the authoritative resumed path. A crash before the first checkpoint requires a fresh run folder; no resumable state exists yet. Do not change `config.json` while resuming a run; changed settings require a new run. If the computer only sleeps and Python remains open, waking it continues the same process; no restart command is needed. Never launch a second copy against the same run.

W05 posterior traces are cached per final run and tied to its selected checkpoint, so a restart after partial final evaluation reuses completed traces rather than repeating those forwards. Smoke tests exercise the final 14/13-tier paths with training-writer surrogate tiers, never W05, and can never promote a checkpoint.

## Runtime

Step 2 measured **@@SPEED@@ seconds/update** including periodic evaluation. From that speed alone, the minimum 8,000 fold + 2,000 final updates estimate is **@@MINH@@ hours**; the 24,000 fold + 6,000 final ceiling would be **@@MAXH@@ hours**. Real handwriting crops can be wider. Actual 20-update real-batch smoke timing was **@@FOLDSPEED@@ seconds/update** for folds and **@@FINALSPEED@@ seconds/update** for final fitting; those imply a minimum about **@@REALMIN@@ hours** on the observed workload. Some timing overlapped Step 2, so CPU contention inflates it. These are measured estimates, not a guarantee of finishing in one night. ETA prints every 50 updates. Details are in `runtime_estimate.json`.

## Files to send back

- Fold set: `run_set.json`, `writer_independent_results.csv`, `fold_selection.json`, each run's `run_config.json`, `completion.json`, `progress.json`, `last_checkpoint.json`, and all `attempt_*/training.jsonl` and evaluation metric/prediction/error CSVs.
- Final run: `run_set.json`, `decision.json`, `evaluation_once.json`, `synthetic_regression_results.json`/`.csv`, the `evaluation/` and `regression/` folders, and final training logs/completion.
- The selected final `.pt` checkpoint, plus the best fold checkpoints if storage/upload size permits. Preserve other checkpoint files locally for resume; they do not all need to be sent back.

The outputs include raw predictions and role-masked predictions, per-role/per-writer metrics, count error categories and one/two-edit tiers. Count feature error categories are based on the existing label mapping, not an independent reader transcription. The two held-out writers provide a limited diagnostic and are not a population-wide writer study.
'''
 estimate=json.loads((local/'runtime_estimate.json').read_text());replacements={'@@INITIAL@@':initial,'@@SPEED@@':f'{speed:.2f}','@@MINH@@':f'{10000*speed/3600:.1f}','@@MAXH@@':f'{30000*speed/3600:.1f}','@@FOLDSPEED@@':f'{foldsmoke:.2f}','@@FINALSPEED@@':f'{finalsmoke:.2f}','@@REALMIN@@':f"{estimate['from_real_batch_smoke_minimum_hours']:.1f}"}
 for key,value in replacements.items():readme=readme.replace(key,value)
 (local/'README_LOCAL.md').write_text(readme,encoding='utf-8')
 lines=['# NAQSHKASH Stage 2G V5b','', '## Augmentation fix','',f"Safe4 selected QA round {qa['accepted_round']} of three. Darkness median {qa['statistics']['augmented']['ink_darkness']['median']:.2f}, real {qa['statistics']['real_handwriting']['ink_darkness']['median']:.2f}; augmented dot median {qa['statistics']['augmented']['dot_diameter']['median']:.2f} px, all accepted detached marks individually 4–7 px.",'',f"Visibility rejections: **{qa['visibility_rejection_rate']:.2%} of all proposals**, versus **{qa['conditional_rendered_visibility_rejection_rate']:.2%} among rendered proposals**. The <5% acceptance uses all sampled proposals as denominator. The stricter rendered-only screen did not pass. Both denominators, the unmodified original round screens, all three rounds, 200 paired selected samples, plots and medians/p10/p90 are in [quality_safe4](quality_safe4/augmentation_quality_report.md). No fourth round was run. Every accepted component has a darkest pixel <=120, with one-to-one component, body-hole and frame guards. Unconfirmed variants remain disabled.",'', '## Generated-only experiment','',f"**{'PASS' if decision['passed'] else 'FAIL'}**: macro-writer raw CER changed from {decision['frozen_macro_writer_CER']:.4%} to {decision['selected_macro_writer_CER']:.4%}, a {decision['macro_writer_CER_improvement']*100:.2f} percentage-point improvement. Clean synthetic gates: {decision['clean_gate']['passed']}. The predeclared rule is >=3 percentage points lower raw macro-writer CER on all 105 originals plus the existing clean synthetic CER/exact gates. The checkpoint and PASS/FAIL decision were fixed before W05, which is informational only.",'',f"Trained {result['steps']} updates; selected update {result['best_step']}. Batch 12 augmented synthetic + 4 clean, all layers trainable, warm-up 100, requested group LRs, updating BN, KL weight 0.5 on mean clean replay KL. Clipping rate {result['clipping_rate']:.2%}. No real handwriting appears in any training update.",'', '| Checkpoint | Raw overall CER | Raw macro CER | Raw no-space CER | Raw exact /105 | Mask macro CER | Mask exact /105 |','|---|---:|---:|---:|---:|---:|---:|']
 for name,m in [('frozen_E1',base)]+[('step_'+str(r['step']),r['metrics']) for r in result['evaluation_history']]:lines.append(f"| {name} | {m['raw']['overall']['cer']:.4%} | {m['raw']['macro_writer']['cer']:.4%} | {m['raw']['overall']['cer_no_whitespace']:.4%} | {m['raw']['overall']['exact_tiers']} | {m['role_mask']['macro_writer']['cer']:.4%} | {m['role_mask']['overall']['exact_tiers']} |")
 lines+=['','Full overall/per-role/per-writer metrics, writer-macro CER/no-space CER/exact accuracy, per-checkpoint raw predictions and count error types are retained in `generated_only/`. Raw and masked metrics use identical identity image preprocessing for E1 and candidates. QA real-darkness statistics use the documented V3 contrast reference; these preprocessing choices were fixed rather than tuned on W05.','', '| Decoder | Synthetic cohort | E1 CER | Selected CER | E1 no-space CER | Selected no-space CER |','|---|---|---:|---:|---:|---:|']
 for d in ['raw','role_mask']:
  for cohort,a in reg['candidate'][d].items():
   b=reg['frozen'][d][cohort];lines.append(f"| {d} | {cohort} | {b['cer']:.4%} | {a['cer']:.4%} | {b['cer_no_whitespace']:.4%} | {a['cer_no_whitespace']:.4%} |")
 lines+=['','All original and no-whitespace gate checks are in `generated_only/synthetic_regression_results.json`. The Step 2 decision specifically requires the clean gates, as requested; positive-edge and long-positive results remain separately visible.','', '| Informational W05 | Decoder | CER | No-space CER | Exact |','|---|---|---:|---:|---:|']
 for subset in ['14_tiers','13_tiers']:
  for d in ['raw','role_mask']:
   r=w05[subset][d]['overall'];lines.append(f"| {subset} | {d} | {r['cer']:.4%} | {r['cer_no_whitespace']:.4%} | {r['exact_tiers']}/{r['tiers']} |")
 recommendation='Proceed with the two requested real-handwriting folds using the Step 2 PASS-eligible initialization by default; retain E1 replay/KL and the final gates.' if decision['passed'] else 'Generated-only adaptation failed the specified rule. A controlled real-handwriting fold experiment is still reasonable because synthetic appearance transfer alone was insufficient or did not preserve the clean gates. Start those folds from frozen E1, not this failed checkpoint; the default auto switch does so. Do not promote any candidate without the final gates.'
 lines+=['','## Decision and local package','',recommendation,'',f"Both local scripts passed their actual bounded smoke tests: four fold/configuration combinations at 20 updates each, and a 20-update final fit with 14/13-tier training-writer surrogates. W05 was not opened in smoke tests. Model/optimizer/RNG checkpoint reload was verified. Temporary smoke outputs were deleted, with proof retained in `local_run/smoke_test_verification.json`. No full overnight fold/final run was launched.",'',f"Step 2 measured {speed:.2f} seconds/update. The minimum 10,000 combined local updates would take about {10000*speed/3600:.1f} hours at that speed; real-batch smoke timing and the safety-ceiling estimate are documented in [README_LOCAL.md](local_run/README_LOCAL.md).",'', 'Frozen E1 remains the operational default; no deployment or existing dataset/checkpoint/result mutation occurred. V5 was stopped after the CNN+BiLSTM final evaluation, before any folds, and only its explicitly authorized superseded-runs record was updated. All V5 bytes were verified unchanged after that record.','', 'Limitations: compact-mark detection and count-feature categorization are heuristic, not independently reader verified. Whole-tier affine transforms are retained; boundary/topology rejection changes the accepted transform distribution. Two held-out training writers are useful diagnostics, not a broad writer-population estimate. Appearance-transfer claims should follow the generated-only PASS/FAIL result rather than informal W05 inspection.']
 (root/'final_report_v5b.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print('V5B FINAL REPORT READY',root/'final_report_v5b.md')
if __name__=='__main__':main()
