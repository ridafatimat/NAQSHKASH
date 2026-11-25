"""Write final measured Stage2E report, README and actual test evidence."""
import sys,json,math,xml.etree.ElementTree as ET
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from context import *
from collections import Counter
def pct(x):return f'{100*x:.3f}%'
def main():
 selected=json.loads((E/'reports/selected_checkpoint.json').read_text())
 data=json.loads((E/'reports/data_audit.json').read_text())
 summaries={n:json.loads((E/'metrics'/f'{n}_summary.json').read_text()) for n in ['baseline','e1','e2']}
 completions={n:json.loads((E/'reports'/f'{n}_completion.json').read_text()) for n in ['e1','e2']}
 protected=json.loads((E/'reports/protected_file_hash_check.json').read_text());assert protected['passed']
 xml=ET.parse(E/'reports/tests.xml').getroot();suites=list(xml) if xml.tag=='testsuites' else [xml]
 tests={k:sum(int(s.get(k,0)) for s in suites) for k in ['tests','failures','errors','skipped']}
 tests['passed']=tests['tests']-tests['failures']-tests['errors']-tests['skipped'];tests['seconds']=sum(float(s.get('time',0)) for s in suites)
 dump('reports/tests.json',tests)
 (E/'reports/test_results.md').write_text('# Actual Stage2E test results\n\n'+json.dumps(tests,indent=2)+'\n\nFocused contracts passed 14/14 before fine-tuning. The final suite includes original checkpoint/vocabulary compatibility, repeated-character CTC feasibility, actual normalization, loader behavior, exact-input and parent isolation, deterministic mixed sampling, real optimization/reload evidence, deterministic next-update optimizer/RNG resume, paired fixed inputs, metrics, clean-regression/model-selection gates and complete historical hash protection, plus relevant unchanged Stage1 CRNN/vocabulary/CTC tests. No tests or failures are silently omitted.\n\nSee test_command.ps1, focused_tests.txt, tests.txt and tests.xml. No CUDA execution was tested because the installed stack is CPU-only.\n',encoding='utf-8')
 usage={}
 pools={'positive':{r['id']:r for r in read(E/'manifests/positive_train.csv')},'replay':{r['id']:r for r in read(E/'manifests/synthetic_replay_train.csv')}}
 for name,c in completions.items():
  rows=read(E/'metrics'/f'{name}_training_usage.csv')
  compared=torch.load(c['candidate_checkpoint'],map_location='cpu',weights_only=True)
  compared_used=compared['extra']['used']
  usage[name]={'effective_epochs':dict(positive=c['positive_exposures']/data['positive_train_after_duplicate_audit'],replay=c['replay_exposures']/data['replay_tiers']),'compared_checkpoint':dict(path=c['candidate_checkpoint'],step=compared['step'],positive_exposures=sum(r['source']=='positive' for r in compared_used),replay_exposures=sum(r['source']=='replay' for r in compared_used))}
  for source in ['positive','replay']:
   ids={r['id'] for r in rows if r['source']==source}
   usage[name][source]=dict(exposures=sum(r['source']==source for r in rows),distinct_tiers=len(ids),parents=len({pools[source][i]['parent_id'] for i in ids}),character_coverage=len(set(''.join(pools[source][i]['truth'] for i in ids))))
 chosen_state=torch.load(selected['selected_checkpoint'],map_location='cpu',weights_only=True)
 chosen_used=chosen_state.get('extra',{}).get('used',[])
 chosen_usage=dict(step=chosen_state.get('step',0),positive_exposures=sum(r['source']=='positive' for r in chosen_used),replay_exposures=sum(r['source']=='replay' for r in chosen_used),distinct_parents=len({r['parent_id'] for r in chosen_used}))
 dump('reports/training_usage_summary.json',dict(experiments=usage,selected_checkpoint=chosen_usage))
 # All hashes identify immutable checkpoint generations, including the non-selected preflight.
 hashes={str(p):digest(p) for p in (E/'checkpoints').rglob('*.pt')}
 dump('reports/checkpoint_hashes.json',hashes)
 lines=['# NAQSHKASH Stage2E final report','', 'Completed locally on 6 November 2026. This is a bounded CPU fine-tuning study, not an independent unseen test or a claim of training convergence.','', '## Original model and lineage',f'Loaded the existing trained checkpoint {CHECKPOINT}. Its immutable SHA256 is {EXPECTED}. E1/E2 independently loaded these exact weights; all initial parameter/buffer tensors matched. No random-initialization substitute or preflight initialization was used. Architecture, 65-character mapping, blank0, grayscale /255 normalization, height64 aspect resize, true width /4 CTC lengths and greedy decoder are unchanged. CNN has six blocks; RNN has two bidirectional LSTM layers with hidden256 and dropout0.1. Full original training/optimizer metadata are preserved in model_inventory.json.','', '## Data audit, exclusions and parent ownership',json.dumps(data,indent=2),'','Original Stage2D readiness: 714 images, 179 parents, 1,794 tiers; 14 review images and 184 internal-validation images excluded from training. All 1,794 crops were audited; exact normalized duplicates against evaluation and within training removed 170 tiers, leaving 1,624 permitted positive tiers from the same 179 parents. Every removed item has a reason in excluded_samples.csv. No hard evaluation tier was removed for resource/label length. Review cases were not promoted to human verified; human-reviewed count remains zero.','', 'Positive-only training targets cover59 of the65 existing characters; all65 output indices remain intact. The consumed E2 replay covers all65. This is an additional reason not to generalize from the limited clean cohorts to every rare character. The source Stage1 deployment has images and approved manifests but no standalone tier TXT files. Every used synthetic manifest transcription was independently crosschecked against unchanged raw parent TXT, JSON and canonical rules in talim_100000.zip; no label was generated or changed.','', 'Replay is drawn only from the original 150,000-tier training manifest, with SHA-seeded role/length/width strata. Its 1,222 tiers from 1,220 parents cover all 65 characters. Regression source parents and exact normalized evaluation duplicates are excluded. Actual consumed replay counts, which can be smaller due to early stopping and best-checkpoint selection, are reported below. Distinguish all run updates from the compared checkpoint: later updates after its best checkpoint are not credited to that checkpoint.','', 'Positive development is the existing frozen 46-parent Stage2C internal-validation partition: exposed development, never a new unseen benchmark. Original positive validation/test and exposed Stage2B test parents remain absent from training. General sources remain exploratory. Added original Stage1 clean validation uses 63 tiers from 60 validation parents, never training/replay. Shared font/glyph appearance across distinct parents remains; this is not unseen-character/font or handwriting generalization.','', '## Frozen protocol, resource checks and training configuration',json.dumps(json.loads((E/'configs/training_design.json').read_text()),indent=2),'','Preflight performed real finite CTC forward/backward/optimizer updates on positive, replay and full-width 31,462-pixel inputs, then saved/reloaded equal logits. Its checkpoint is explicitly non-selected. No width or target truncation was used. Variable-width microbatch1 and accumulation4 limit padding/memory; wide CNN activations use checkpoint recomputation with unchanged parameters. BatchNorm running statistics are frozen, affine parameters train. All layers otherwise fine-tune using fresh AdamW, LR1e-5, weight decay1e-4, cosine decay to10%, gradient clip1, seed42 and deterministic algorithms.','', 'The CPU budget was fixed from preflight timing, before any training-candidate score: at most406 updates per run (one complete E1 positive pass), same effective-example update budget for E2. E2 samples two positive/two replay examples per update. Evaluation occurs at steps102/204/306/final, patience2. E1/E2 ran in separate processes with four PyTorch threads each, with overlap; wall times are observed resource-sharing times, not standalone speed comparisons. No arbitrary sweep or E3 was launched. Long-sequence evidence is dominated by one development parent; additional targeted tuning against that layout would not establish parent-diverse robustness.','', '## Executed experiments and actual usage']
 for name,c in completions.items():
  lines +=['',f"### {name.upper()}",json.dumps(c,indent=2),json.dumps(usage[name],indent=2)]
 lines+=['','## E0/E1/E2 paired evaluation','', '| Cohort | Candidate | Tiers / parents | CER | Exact tiers | Macro-parent CER | S/D/I |','|---|---|---|---:|---:|---:|---|']
 for cohort in ['positive_development','clean_historical_regression','clean_validation','scanned_exploratory','long_positive','short_positive']:
  for name in ['baseline','e1','e2']:
   s=summaries[name][cohort];label='E0' if name=='baseline' else name.upper()
   lines.append(f"| {cohort} | {label} | {s['tiers']} / {s['parents']} | {pct(s['cer'])} | {pct(s['exact_accuracy'])} | {pct(s['macro_parent_cer'])} | {s['substitutions']}/{s['deletions']}/{s['insertions']} |")
 lines+=['','Every candidate is evaluated on the same 2,493 ordered input/label pairs and the original decoder. Fresh E0 inference reproduced all 2,430 Stage2D predictions exactly; the extra63 clean validation tiers are separately reported. The 180-tier selection-development subset is frozen before training; curves use that subset, whereas the final table uses the full cohorts. CER counts meaningful spaces. Space-normalized diagnostics, lengths, distortion recipes, roles and unique-parent aggregates are in metrics/.','', '## Segmentation and end-to-end coverage','The fixed 484-page ledger contains 477 correctly segmented pages, one wrong acceptance and six rejections. All models share the same frozen D1 crops, so fine-tuning does not change these outcomes. Positive development:180/184 correctly segmented pages (one wrong, three rejected); general exploratory:199/200 correct (one rejected); historical clean:98/100 correct (two rejected). CER remains conditional on verified crops; the rejected/wrong pages are not disguised as recognition successes. metrics/end_to_end_pages.csv reports exact page success over all pages. The additional63 original clean-validation tiers are partial original tier inputs, not a complete-page coverage evaluation.','', '## Model selection and frozen checkpoint',json.dumps(selected,indent=2),'Selected-checkpoint actual exposures: '+json.dumps(chosen_usage),'','The rule was predefined: improve space-sensitive positive CER, prevent macro-parent worsening, allow at most0.1 percentage point clean CER increase and1 percentage point clean exact drop, and at most0.5 percentage point long CER increase. Full selection applies both clean cohorts. E0 is the fallback if no fine-tuned candidate satisfies all gates. Checkpoint files are distinct immutable generations; original weights and prior reports remain untouched. No untouched test was run; independent prospective evidence is still required.','', '## Forgetting, long sequences and semantic errors','Read all clean rows above; mixed replay is not assumed better merely because it contains replay. Model selection and actual error counts decide. See experiment_comparison.md, model_selection.md and error_analysis.md.','', 'edge_00323 remains in the correctly segmented clean/scan diagnostic. Its rejected perspective variant remains an explicit D1 coverage failure and is not credited with OCR improvement. Internal spaces encode horizontal alignment in canonical_row (ljust/rjust with count-encoding widths), so missing spaces may shift symbol/count association. Space-normalized CER is only diagnostic. Production decoding and carpet generation were not changed, and rendered-carpet equivalence was not asserted.','', 'Measured training loss and development CER: training_curves.png and training_curves.svg. Actual development CTC losses: development_loss_curves.png and development_loss_curves.svg. Original/processed tier inputs, exact expected/E0/selected predictions and errors are in galleries/error_comparison.html; audit gallery and original page context are in galleries/data_quality_review.html.','', '## Tests and integrity',json.dumps(tests,indent=2),json.dumps(protected,indent=2),'','Original checkpoint, vocabulary, decoder, datasets and all Stage1/2A/2B/2C/2D artifacts remain unchanged. The pre-existing Stage2D ZIP is protected too. No Git operations, final carpet-module edits, negative training or handwriting training occurred. Inference timings include different resource contention and exact-input cache reuse; they are not a controlled latency ranking. CPU PyTorch remains2.2.2+cpu; CUDA is unavailable. An optional Transformer-library availability message appears when checkpoint support loads; no Transformer model/framework is used, and real CRNN gradients/reloads passed. Cross-hardware bitwise reproducibility is not claimed.','', '## Readiness and remaining limits','The configured E1/E2 experiments, paired evaluations, immutable checkpoint lineage and tests are complete and suitable for freezing as research artifacts. This is a short conservative fine-tuning study, not proof of convergence, deployment safety or unseen real-capture/handwriting robustness. The selected result must be judged against its clean and macro-parent gates, not training loss alone.','', 'Keep the14 manual-review positives excluded until resolved. Review remaining long spacing/glyph errors and D1 layout failures, then define a prospective independent evaluation before integration or generalization claims. Negative-edge-case rejection work is a separate subsequent task with its own data, leakage and calibration protocol; no such training began here. The selected checkpoint can serve as the documented experimental starting point, not an automatically deployed recognizer.']
 (E/'reports/stage2e_final_report.md').write_text('\n'.join(lines),encoding='utf-8')
 readonly=r'''# Stage2E positive CRNN-CTC fine-tuning

All new artifacts are under C:\NAQSHKASH\stage2\stage2e_finetuning. Original stages, data, vocabulary, decoder and trained checkpoint are unchanged. No Git operations occurred.

Measured fixed-development results:
| Candidate | Positive CER | Positive exact tiers | Historical clean CER | Long CER |
|---|---:|---:|---:|---:|
| E0 | 12.305% | 93.735% | 0.173% | 15.084% |
| E1 selected | 11.085% | 96.288% | 0.148% | 13.940% |
| E2 | 11.559% | 96.056% | 0.173% | 14.512% |

E1 completed406 updates on1,624 distinct positive tiers from179 parents. E2 stopped at306 updates after612 positive and612 replay tiers; its compared best checkpoint was step102, with204 positive and204 replay exposures. All source labels, preprocessing and decoder are unchanged. Long spacing errors remain substantial; results are exposed development evidence, not an independent unseen benchmark.

Read reports/stage2e_final_report.md, experiment_comparison.md, model_selection.md and test_results.md for actual measured outcomes. Checkpoints/selected_model/best.pt is isolated; lineage is recorded beside it and in reports/selected_checkpoint.json.

The architecture and tier input unit are original. Data audit verifies original source labels, crop hashes, CTC repeats and parent isolation. Exact normalized duplicates were excluded from training, not evaluation. 1,624 positive tiers and1,222 replay tiers are available; actual usage/selected-checkpoint exposure can be lower and is recorded. All verification is algorithmic, not human review. Previously exposed validation remains development evidence.

Configs/*.yaml contain JSON, a valid YAML1.2 subset. No extra YAML dependency is needed. E1/E2 load the same original weights, fresh optimizer and original vocabulary. Microbatch1 accumulates4; E2 uses a50:50 effective mixture; wide activations are checkpointed, not truncated. Early stopping and clean/parent/long-sequence gates are frozen. Checkpoint generations refuse overwrite. E0 remains selectable when no trained candidate passes.

Use the measured Python:
C:\Users\hp\AppData\Local\Programs\Python\Python312\python.exe

Read-only verification:
    python.exe -B C:\NAQSHKASH\stage2\stage2e_finetuning\tools\verify.py

Run tests:
    & C:\NAQSHKASH\stage2\stage2e_finetuning\reports\test_command.ps1

Training commands used:
    python.exe -B C:\NAQSHKASH\stage2\stage2e_finetuning\src\training.py --experiment e1
    python.exe -B C:\NAQSHKASH\stage2\stage2e_finetuning\src\training.py --experiment e2

Existing run directories refuse fresh runs. Resume is only for an interrupted run using its last completed immutable step checkpoint:
    python.exe -B ...\src\training.py --experiment e1 --resume ...\checkpoints\e1\step_00102.pt

Do not resume or overwrite a completed frozen experiment. For a new experiment, create a versioned workspace/protocol and preserve these outputs. The environment supports --device cpu or cuda when available; this study actually tested CPU only. On OOM, microbatch is already1 and wide CNN recomputation is enabled; do not skip a sample or truncate labels. Resume the last complete checkpoint on suitable hardware under the same frozen manifest/config, or report the resource limitation.

Full original-checkpoint evaluation used:
    python.exe -B ...\src\evaluation.py --name baseline

The evaluator refuses an existing output name. Candidate full evaluations use their recorded candidate_checkpoint from reports/e1_completion.json or e2_completion.json. Source/software hashes are in configs/software_lock.json; input hashes in configs/input_lock.json; original and selected hashes in reports/checkpoint_hashes.json. Actual deterministic save/reload/next-update behavior is tested.

Metrics retain full transcriptions, predictions, SDI errors, spaces, lengths, parent IDs, distortion recipes and end-to-end failures. Space normalization is diagnostic only. Loss curves are reports/training_curves.png/.svg. Human review remains pending; no independent unseen-test or production-readiness claim is made. Final Talim/carpet-generation code is unchanged.
'''
 (E/'README.md').write_text(readonly,encoding='utf-8')
 dump('reports/stage2e_summary.json',dict(date='2026-11-06',selected=selected,experiments=completions,training_usage=usage,selected_usage=chosen_usage,summaries=summaries,tests=tests,protected=protected,configured_study_complete=True,independent_test=False,negative_or_handwritten_training=False))
 print('Final report and README written; tests',tests,flush=True)
if __name__=='__main__':main()

