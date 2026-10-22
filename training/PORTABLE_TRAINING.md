# Stage 1 single-GPU training

For Kaggle/Colab notebooks, ZIP handling, GPU diagnostics, the controlled 20+20
batch resume test and persistence, see [GPU_DEPLOYMENT.md](GPU_DEPLOYMENT.md).
Notebook Run All never starts real full training.

Run commands from the repository root with Python 3.8+ and PyTorch 2.3+
(keep the platform's CUDA-enabled PyTorch). Existing requirements otherwise apply.
The model, label encoding, vocabulary, right padding, full epoch validation,
CER selection, scheduler, early stopping and final test flow are preserved.

## Batching and memory preflight

The default width budget is **16384 resized width units**, maximum 32 samples.
At height 64 this bounds padded image area to 1,048,576 pixels. This is a
conservative starting hypothesis equivalent to 16 images of width 1024, **not a
measured Tesla T4 capacity**. Original run widths are unknown. Run the GPU probe
and lower the budget if needed. A single image wider than the budget causes an
explicit error; no sample is dropped. Increase the budget if such images exist,
then run the probe. CNN activations and recurrent workspace dominate raw pixels.

Each epoch shuffles sample indices with a private Python RNG using seed + epoch,
sorts widths inside shuffled pools of 2048, greedily packs within the width budget
and maximum batch size, then shuffles final batch order. Every sample appears
exactly once. Validation and final test also use bounded batches. Their sample
metrics remain full-split metrics; mean batch loss can differ from fixed batching.

PNG dimensions are read from headers once and cached in the output folder.
Other image formats are decoded only when populating the cache. Metadata/sample
identities and image size/mtime validate cache reuse. Dataset files must remain
immutable during training. The sampler checkpoint stores ordered widths and
metadata identity to detect incompatible reconstruction across platform mounts.

Startup checks the largest padded-area batch and the longest-sequence batch with
forward, CTC loss, backward, gradient clipping and an AdamW step. Model buffers,
optimizer and RNG states are restored afterward. It does not advance training.
OOM raises an actionable error. This is a useful probe, not a guarantee against
every CUDA allocator/workspace condition. Future epoch grouping can vary within
the same bound. Leave GPU memory headroom.

## AMP and resume

`--amp` enables CUDA autocast only around the model forward pass. Log-softmax and
CTC operate in FP32. GradScaler scales backward, unscales before clipping,
steps AdamW and updates its scale. CPU runs use FP32. The scaler state is saved.
ReduceLROnPlateau steps once after full validation using validation CER.

`last.pt` is atomically replaced every 1000 training batches by default; `best.pt`
is replaced only on validation CER improvement. A failed temporary write leaves
the previous final file intact. There is no rotating checkpoint series.

Checkpoint epochs are zero based: epoch=0,next_batch=30001 reconstructs epoch 0
and slices its index plan at batch 30001 without loading preceding images.
At epoch end last.pt points to epoch+1,next_batch=0. It retains global step,
partial-epoch loss totals, history, early stopping, scheduler, optimizer, scaler,
exact vocabulary mappings, architecture/configuration, sampler identity and
Python/CPU/CUDA RNG states. DataLoader uses a separate RNG, so constructing its
iterator on resume does not consume model RNG. Current dataset has no random
transforms; worker-side random augmentation would require additional state.
CUDA operations are not promised to be bitwise deterministic across devices.

Resume rejects incompatible vocabulary, architecture, limits, seed, budgets,
AMP configuration or sampler metadata/widths. Paths, worker counts and total
epoch target may change. Legacy epoch-only checkpoints are rejected with a clear
message. Resume after validation needs the matching best.pt alongside last.pt;
before the first validation, only last.pt is needed. Copy these files and width
caches to persistent storage before an ephemeral Colab/Kaggle runtime ends.
Atomic writes cannot persist storage that the hosting platform deletes.

## Commands (Linux / Colab / Kaggle)

Set these shell variables to existing paths. Colab example:

```bash
DATASET=/content/stage1
RUN=/content/drive/MyDrive/NAQSHKASH/stage1_run
```

Kaggle example (replace the input dataset slug):

```bash
DATASET=/kaggle/input/your-stage1-dataset/stage1
RUN=/kaggle/working/stage1_run
```

GPU memory preflight scans the full training widths and probes the expensive
batches without running an epoch, validation or test:

```bash
python -m training.train_crnn --dataset-root "$DATASET" --output-root "$RUN" --width-budget 16384 --max-batch-size 32 --amp --num-workers 2 --preflight-only
```

Full training (30 maximum epochs, early stopping retained):

```bash
python -m training.train_crnn --dataset-root "$DATASET" --output-root "$RUN" --epochs 30 --width-budget 16384 --max-batch-size 32 --amp --num-workers 2 --checkpoint-every-batches 1000
```

Resume (same configuration, portable dataset/output paths):

```bash
python -m training.train_crnn --dataset-root "$DATASET" --output-root "$RUN" --epochs 30 --width-budget 16384 --max-batch-size 32 --amp --num-workers 2 --checkpoint-every-batches 1000 --resume "$RUN/last.pt"
```

Optional checkpoint smoke test: use a separate output directory and append
`--stop-after-batches 5 --skip-test`. It saves last.pt and exits without validation
or test. Resume that directory with identical options except stop-after-batches.
`--batch-size` remains supported as a fallback maximum if --max-batch-size is
omitted. --train-limit and --val-limit remain available for tiny smoke tests;
they are not used in the full-run commands above.

## Verification

The automated suite includes deterministic coverage/budget/resume checks, width
cache checks, atomic-write interruption recovery, optimizer/scheduler/scaler
serialization, and an actual CRNN subprocess interruption/resume before first
validation followed by final model selection/test. A CUDA scaler test runs only
when CUDA is available. Local hardware is CPU-only; GPU capacity and CUDA AMP
must be verified with the preflight on the target GPU.
