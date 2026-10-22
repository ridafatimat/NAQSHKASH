# Stage 1 T4 numerical diagnostic benchmarks

Follow-up: the T4 diagnostic found zero non-finite losses and seven non-finite
gradient batches during initial scale backoff, with no further failures after
approximately step 35 through step 200. Fresh manifest-training AMP runs now use
--amp-init-scale 4096 by default (previously PyTorch's implicit 65536). Override
the fresh-run value with --amp-init-scale VALUE; it must be positive and finite.
Growth factor, backoff factor, growth interval and update logic remain unchanged.
Resume still calls scaler.load_state_dict on the checkpoint: saved scale and
growth/backoff state take precedence over the CLI initialization, including when
resuming older v3 checkpoints that have no amp_init_scale config field. This
fresh-run setting is deliberately not a resume compatibility constraint.

The first real T4 benchmark reported approximately 11.4 optimizer steps/sec,
256.1 samples/sec, average batch size 22.47, 0.25% data wait, 1224 MiB peak memory,
and seven skipped batches by step 200. No packing, copying or loader changes are
justified or implemented. No model, vocabulary, manifest, encoder or scheduler
design changes are included in this diagnostic patch.

Previously the skip condition was:

    not bool(torch.isfinite(loss) & torch.isfinite(norm))

Here norm is the total unscaled L2 gradient norm returned by clip_grad_norm_.
Either NaN/Inf loss OR NaN/Inf norm triggered the same warning after backward.
The old logs cannot establish which operand failed, let alone the underlying
GPU numerical cause. Seven skips alone do not establish corrupt data or an AMP
bug. AMP backward overflow is a hypothesis to test, not an established diagnosis.

Loss is now checked independently before backward. Non-finite loss increments
nonfinite_loss_batches and skips backward/optimizer/scheduler; no scaler update
is performed because no backward/unscale inf check occurred. Finite loss proceeds
through scaled backward, unscale, then clipping at 5 as before. A non-finite norm
increments nonfinite_gradient_batches, skips optimizer/scheduler and retains the
existing scaler.update behavior following unscale. No NaN clamp, threshold
increase, optimizer/scheduler change, or permanent AMP disable is introduced.
The cumulative failure threshold remains TEN.

The first five failures across both categories are logged as compact JSON to
stdout and numerical_failures.jsonl (append/flush). Each includes completed global
optimizer step, zero-based epoch, one-based consumed batch index, batch size,
image-width/input-length/target-length/required-timestep min/max, scalar loss,
pre-update AMP scale, logits/log_probs finite flags, log_probs dtype, length
devices/dtypes, total gradient norm when applicable and tier distribution.
NaN/Inf scalars use strings to keep JSON valid. Subsequent failures still warn
and increment counters. --failure-detail-limit defaults to 5. Normal reports and
new metrics.csv files include both counters. Checkpoint v3 adds diagnostic state
only; existing model/optimizer/scaler/scheduler and sampling fields are unchanged.
Older v3 resumes default missing counters to zero and warn about unclassifiable
historical skips; existing CSV headers/history are preserved, while new counters
remain available in stdout/checkpoints. Use fresh output directories for A/B tests.

The forward remains under autocast when AMP is active. A disabled-autocast region
explicitly computes logits.float().log_softmax(2) and calls CTCLoss on FP32 data
with zero_infinity=True. Flattened targets retain the encoder's torch.long indices
and transfer to the model device. Both length tensors remain CPU torch.long for
PyTorch's regular CTC path. We do not force cuDNN's different int32/equal-length
fast path. Every actual batch checks that sum(target_lengths) matches flattened
targets and that output_lengths(actual collated widths) covers encoded target
length plus adjacent repeat count, and does not exceed the actual logits time
dimension. Infeasible data raises; labels are never edited. CPU tests inspect the
actual criterion argument dtype even inside an enclosing autocast region.

Reference: [CTCLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html)
and [AMP examples](https://docs.pytorch.org/docs/stable/notes/amp_examples.html).
Installed PyTorch 2.4.1 CTCLoss/GradScaler/clipping source was also inspected.

Run from the UPDATED code root. Set DATASET_ROOT to the actual attached compact
package root containing manifests/, train/images/ and validation/images/ (include
the top-level package folder if Kaggle extraction retained it). The dataset slug
has not been supplied, so it must not be guessed. Use separate empty output roots.
The defaults inspected are width-budget 16384 and max-batch-size 32; these commands
make both explicit. If the original Kaggle launch overrode them, use its recorded
config values for the A/B baseline instead.

## A: current budget, AMP diagnostic

```bash
python -m training.train_crnn --dataset-root "$DATASET_ROOT" --output-root /kaggle/working/naqshkash_stage1/diag_amp --benchmark-steps 200 --warmup-steps 500 --width-budget 16384 --max-batch-size 32 --num-workers 4 --seed 42 --amp --failure-detail-limit 5
```

## B: AMP-OFF comparison if A still fails

```bash
python -m training.train_crnn --dataset-root "$DATASET_ROOT" --output-root /kaggle/working/naqshkash_stage1/diag_fp32 --benchmark-steps 200 --warmup-steps 500 --width-budget 16384 --max-batch-size 32 --num-workers 4 --seed 42 --no-amp --failure-detail-limit 5
```

Same seed, initial model, manifests, batch plan, LR/schedule and workers. Resume
neither benchmark. Runs are comparable but not numerically identical; after a skip
optimizer progress and AMP/FP32 weights can diverge. Correlate failures by epoch/
consumed batch index. A benchmark targets 200 successful optimizer updates, so
with skips it can consume more than 200 batches; ten failures still abort earlier.
The 500-step warmup matches the existing default despite this short benchmark;
if the first Kaggle run used a different warmup, reproduce it identically in A/B.

Interpretation: finite loss/logits/log_probs with only gradient-norm failures in A
and none in B supports AMP/backward overflow as the cause. Look at failure scales
and whether skips cluster initially or persist. It does not prove which layer or
kernel produced the non-finite value. Forward non-finite logits/log_probs, non-finite
loss in both runs, or a runtime feasibility exception require separate investigation.
Do not begin production merely because the benchmark reaches step 200.

## C: throughput only after numerical stability is confirmed

```bash
python -m training.train_crnn --dataset-root "$DATASET_ROOT" --output-root /kaggle/working/naqshkash_stage1/throughput_2x --benchmark-steps 200 --warmup-steps 500 --width-budget 32768 --max-batch-size 64 --num-workers 4 --seed 42 --amp --failure-detail-limit 5
```

Budget A is 16384 padded width units / 1,048,576 height-64 pixels with cap 32.
Budget C is 32768 / 2,097,152 pixels with cap 64. Defaults are unchanged. Compare
samples/sec, average batch size, skip counts and peak allocated memory; smaller
step/sec can still mean better sample throughput. The measured 1224 MiB peak in
200 steps does not bound every future batch or total reserved GPU memory. This is
a cautious 2x trial, not a guaranteed memory estimate. Stay on one GPU.

Local verification is CPU-only; active T4 AMP failure reproduction remains a
Kaggle task. No real training was started by this patch.

Verification: 22 passed, 1 skipped in 62.03s using PyTorch 2.4.1+cpu. Command:

```bash
python -m pytest tests/test_training_numerics.py tests/test_manifest_training.py tests/test_training_portable.py tests/test_width_resume.py tests/test_ctc_batch.py tests/test_vocabulary.py -q
```

Tests include injected NaN loss and injected NaN gradients: separate counters and
JSON events, no parameter/Adam/scheduler updates, threshold abort and checkpoint
save. Other checks cover first-five logging limits, FP32 criterion input under
autocast, runtime repeated-character feasibility, exact fresh-process resume,
legacy v3 diagnostic defaults and preserving pre-diagnostic CSV headers/history.
Changed training/manifest_training.py and training/train_crnn.py (opt-in checked
validation path); added tests/test_training_numerics.py and this guide. The data
pipeline, architecture, vocabulary, manifests, encoder and scheduler are unchanged.
