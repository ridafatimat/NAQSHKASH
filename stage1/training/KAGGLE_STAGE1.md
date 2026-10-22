# Stage 1 manifest training

Use the new manifest CLI below. The older epoch CLI and notebooks remain available
for compatibility; their full-data/30-epoch commands are not the optimized workflow.
No YAML dependency is needed. Run from the project directory with Python 3.8+ and
PyTorch 2.4+ (retain Kaggle's CUDA PyTorch), NumPy and OpenCV installed.

Inputs: project code including config/vocabulary.json, and the extracted approved
dataset including manifests/stratified_train_150k.csv and
manifests/stratified_validation_10k.csv. Keep relative image paths and filename case.
Code and dataset Kaggle slugs are placeholders until assigned. Change directory to
the actual code root (for example /kaggle/input/naqshkash-code/NAQSHKASH).

Benchmark first, in a fresh output directory:

```bash
python -m training.train_crnn --dataset-root /kaggle/input/naqshkash-data --output-root /kaggle/working/naqshkash_stage1/benchmark_w4 --benchmark-steps 200 --num-workers 4 --amp
```

Repeat with --num-workers 2 and a different output directory (benchmark_w2).
Benchmark performs real optimizer updates but skips validation; never resume a
benchmark into the production schedule. Ignore the initial warmup window when
comparing throughput. Logs report host-visible DataLoader wait and window-level
compute time with CUDA synchronization only at reporting boundaries. Necessary
finite-loss/gradient checks still read GPU scalars for correctness. Data wait is
the training process's exposed stall, not a CUDA profiler measurement. It can
overlap prefetched work. It excludes validation and checkpoint writes.

At <=20% data wait, keep physical files. Consistently >20% justifies investigating
local copying or packing separately; no conversion is implemented here. Compare
steps/sec too, since batch sizes vary. Use the measured optimizer_steps_per_second
from representative windows to set:

    total_steps = floor(0.8 * available_GPU_hours * 3600 * measured_steps_per_second)

The 20% allowance covers validation/checkpoint/startup overhead and interruptions.
Measure those overheads too before committing the final budget. There is no fixed
production total_steps default. Warmup defaults to 500; choose it after benchmarking.

Controlled 300-step save and fresh-process resume (same total schedule both times):

```bash
python -m training.train_crnn --dataset-root /kaggle/input/naqshkash-data --output-root /kaggle/working/naqshkash_stage1/resume_test --total-steps 600 --warmup-steps 50 --stop-after-steps 300 --checkpoint-every-steps 100 --num-workers 4 --amp
python -m training.train_crnn --dataset-root /kaggle/input/naqshkash-data --output-root /kaggle/working/naqshkash_stage1/resume_test --total-steps 600 --warmup-steps 50 --stop-after-steps 300 --checkpoint-every-steps 100 --num-workers 4 --amp --resume auto
```

Compare checkpoint global_step (300 then 600), next_batch/epoch, scheduler last_epoch,
optimizer LR, nonempty Adam state and nonempty T4 scaler state. metrics.csv appends.
For automated exact CPU continuation verification:

```bash
python -m pytest tests/test_manifest_training.py -q
```

After choosing TOTAL_STEPS and WARMUP_STEPS, launch deliberately:

```bash
python -m training.train_crnn --dataset-root /kaggle/input/naqshkash-data --output-root /kaggle/working/naqshkash_stage1/full_run --total-steps "$TOTAL_STEPS" --warmup-steps "$WARMUP_STEPS" --num-workers 4 --amp --checkpoint-every-steps 1000 --validate-every-steps 2000 --max-runtime-minutes 660
```

Use the faster tested worker count. --no-pin-memory, --no-persistent-workers and
--prefetch-factor tune loader behavior; persistent/prefetch settings apply only
with workers >0. --width-budget is maximum padded width times batch size (pixels
are that value times height 64). A sample wider than budget fails without dropping it.

last.pt is atomic and saved periodically and on clean shutdown; best.pt is atomic
on improved fast-validation CER. metrics.csv appends and flushes each report,
including per-tier CER/exact match. Validation is greedy and only uses the approved
10k manifest. Test and full validation are separate final evaluations.

Use --resume auto for output-root/last.pt or --resume PATH for a trusted prior-run
input checkpoint. Paths and worker counts can change; vocabulary, manifest bytes,
limits, batch plan, seed, AMP mode, LR and schedule must match. Save the same
total_steps across chained sessions. A v2 legacy checkpoint cannot resume this v3
path. Download/save Kaggle outputs between sessions and attach the prior output
as input; supply its last.pt path and a new writable output-root. Copy metrics.csv
and best.pt into the new output directory before resuming to retain full history
and the prior best artifact. Checkpoints alone do not contain old CSV rows or the
best model's weights. A surviving CSV may include rows beyond the last completed
checkpoint after an abrupt kill; retain them and deduplicate by global_step for plots.

The time guard reserves 120 seconds for shutdown and checks each training and
validation batch. SIGINT/SIGTERM request a clean stop. It cannot guarantee survival
of a forced kill, hung I/O, or a single unusually slow batch/checkpoint. Early
stopping uses six validation checks without CER improvement. Non-finite batches
skip weight/scheduler updates and stop with an error at ten cumulative failures.

Checkpoint schema v3: model_state_dict, optimizer_state_dict, scheduler_state_dict,
scaler_state_dict; progress plus top-level epoch, next_batch, global_step,
best_val_cer, skipped_batches, stale_validations; random_seed; python_rng_state,
numpy_rng_state (primitive list representation), torch_rng_state, cuda_rng_states;
config, char_to_idx, idx_to_char, vocabulary_hash, manifest_identity, amp_active,
sampler_state (version/seed/pool_size). No unsafe pickle loading is needed.
global_step counts completed optimizer updates; next_batch also includes skipped
invalid batches. RNG is restored after constructing the model/optimizer. Loader
generators are separate from model RNG and data transformations are deterministic.
Exact GPU bitwise reproducibility depends on CUDA kernels; sampler progression is
deterministic. Temp files are fsynced and replaced within the output filesystem.

Never rebuild/reorder vocabulary, change OCR normalization/architecture, alter the
split/test semantics or approved manifests, or modify dataset files. No torch.compile,
distributed training, beam search or packing is used.
