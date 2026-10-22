# Stage 1 optimization results — October 18, 2026

Prepared and tested in the workspace before applying the explicitly listed files
to C:/NAQSHKASH. Existing unrelated project changes are preserved; no commit,
remote operation, Kaggle upload, dataset modification or full training was done.

1. **Changed:** training/train_crnn.py, adding a dispatcher for manifest/step CLI
   arguments. Existing public functions/legacy CLI remain compatible.
2. **Created:** data/manifest_dataset.py; training/manifest_training.py;
   config/vocabulary.json; tests/test_manifest_training.py;
   training/OPTIMIZATION_INSPECTION.md; training/KAGGLE_STAGE1.md; this report.
3. **Loading:** replaced per-fetch TXT reads and all-image startup inventory with
   manifest text/widths, in-memory validation and first-100 image checks per split.
   Approved manifests load exactly 150,000 TRAIN and 10,000 validation samples,
   with zero excluded infeasible samples. Full startup/width planning measured
   **2.95 seconds locally**, widest TRAIN width 4408. No full pixel scan.
4. **Batching:** existing deterministic pool sorting/shuffling, seed+epoch,
   max-batch cap and direct batch offset retained. Approved TRAIN plan contains
   **6608 batches**, maximum padded-width cost **16384**, respecting budget.
5. **AMP/CTC:** preserved forward autocast, float32 log-softmax/CTC and
   zero_infinity=True; CPU length tensors; correct unscale-before-clip order.
   One combined finite-loss/norm safety read per step. Invalid batches do not
   update weights or scheduler; ten cumulative failures abort with checkpoint.
6. **Schedule:** linear warmup then cosine to 0.001 of initial LR, stepped once
   per completed optimizer update. No production total_steps default. Budget
   must be supplied after the benchmark. Validation cadence is resume-critical.
7. **Checkpoint schema:** atomic fsynced v3; model, AdamW, scheduler, scaler,
   epoch/next_batch/global_step/best CER, skipped/stale counts, Python/NumPy/
   Torch/CUDA RNG, config, fixed mapping/hash, manifest hashes, sampler version/
   seed/pool_size and active AMP mode. See KAGGLE_STAGE1.md for exact key names.
8. **Resume:** auto or explicit path, exact schedule/manifest/vocabulary/critical
   settings checks, separate DataLoader RNG and direct sampler offset. Metrics
   append; external-session history/best artifacts need copying as documented.
9. **Portability:** pathlib paths, UTF-8 reads, Linux case preserved, configurable
   workers/pin/persistence/prefetch, explicit runtime guard, signal shutdown.
   CLI-only configuration avoids a YAML dependency. Legacy notebooks still
   describe the old epoch workflow; use the new guide's commands for this phase.
10. **Timing:** reports batch size, samples/sec, optimizer steps/sec, host data
    wait/compute percentages, loss, LR and peak CUDA memory. CUDA timing sync is
    only at log-window boundaries. Validation/checkpoint windows are separate.
    CPU measurements are not evidence for Kaggle I/O/compute bottlenecks.
11. **Tests:** full suite **90 passed, 1 skipped in 70.36s**. Earlier sandbox temp
    errors were resolved by rerunning with filesystem approval. Exact continuation
    integration compared uninterrupted vs fresh-process resumed training at step
    12: identical model tensors, Adam state tensors/parameter groups, scheduler
    and progress; metrics append and incompatible budget rejection passed.
    Real approved-data CPU smoke: 500 TRAIN / 30 validation samples, 50 updates,
    last.pt/best.pt/metrics produced, loss window 63.27 -> 4.26; CER remained 1.0,
    so this is only a pipeline smoke. Tiny eight-image single-label fixture:
    300 updates, 100% exact / CER 0 by step 100; fresh-process resume restored
    step 300 and LR 3.647412130512766e-06, continued to step 310 / LR 1e-06,
    scheduler last_epoch=310, Adam state for 42 parameters, seven preserved CSV
    rows. This fixture checks learning mechanics, not OCR quality. Two-worker
    persistent/prefetch loader smoke: four updates completed. Sampler ordering,
    exclusion, hash changes, unknown symbols and dimension mismatch tested.
12. **Limits:** local PyTorch 2.4.1 is CPU-only. Active CUDA scaler restoration,
    T4 peak memory, throughput, worker 2-vs-4 speed and actual I/O wait remain
    Kaggle checks. Old v2 checkpoints cannot resume into the new schedule.
    Timings include necessary scalar correctness reads and are coarse wall-time
    instrumentation, not a GPU profiler. Abrupt termination can leave CSV rows
    newer than last checkpoint; retain/deduplicate by step. Hard kills/hung I/O
    can exceed the shutdown margin. No packing decision is justified locally.
13. **Exact Kaggle benchmark:**

```bash
python -m training.train_crnn --dataset-root /kaggle/input/naqshkash-data --output-root /kaggle/working/naqshkash_stage1/benchmark_w4 --benchmark-steps 200 --num-workers 4 --amp
```

14. **Exact controlled resume commands:**

```bash
python -m training.train_crnn --dataset-root /kaggle/input/naqshkash-data --output-root /kaggle/working/naqshkash_stage1/resume_test --total-steps 600 --warmup-steps 50 --stop-after-steps 300 --checkpoint-every-steps 100 --num-workers 4 --amp
python -m training.train_crnn --dataset-root /kaggle/input/naqshkash-data --output-root /kaggle/working/naqshkash_stage1/resume_test --total-steps 600 --warmup-steps 50 --stop-after-steps 300 --checkpoint-every-steps 100 --num-workers 4 --amp --resume auto
```

15. **Exact production command after setting budget variables:**

```bash
python -m training.train_crnn --dataset-root /kaggle/input/naqshkash-data --output-root /kaggle/working/naqshkash_stage1/full_run --total-steps "$TOTAL_STEPS" --warmup-steps "$WARMUP_STEPS" --num-workers 4 --amp --checkpoint-every-steps 1000 --validate-every-steps 2000 --max-runtime-minutes 660
```

Replace the dataset slug and change into the actual code root first. Set variables
from representative measured throughput using the 80% GPU-hours budget in the
guide. Use the faster worker count. Do not launch production until measured.
