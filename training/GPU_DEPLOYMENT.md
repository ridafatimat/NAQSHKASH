# Step 3: Kaggle first, Colab backup

No GPU capacity has been measured locally. Run both GPU probes and the controlled
resume test before manually starting real training. Notebook Run All never starts
the real 30-epoch command. Use Python 3.8+ and platform CUDA PyTorch 2.3+.

## Kaggle setup and layout

Upload the prepared NAQSHKASH_GPU_Code.zip and your existing Final_tier_dataset.zip
as separate private Kaggle datasets, attach both, then import
colab/NAQSHKASH_Stage1_Kaggle.ipynb. No GitHub is needed. Enable a GPU accelerator.
The notebook exposes GPU 0 only, even if two GPUs are offered.

Set just three paths near the top (inspect your actual slugs/wrapper directories):

```text
PROJECT_ROOT=/kaggle/input/<code-dataset>/NAQSHKASH
    training/, models/, data/, tests/, colab/, requirements.txt
DATASET_ROOT=/kaggle/input/<stage1-dataset>
    train/, validation/, test/ each containing images/, labels/, metadata.csv
OUTPUT_ROOT=/kaggle/working/naqshkash
    quick_probe/       subset memory test; no .pt
    resume_test/       controlled checkpoint/resume test
    full_width_probe/  full TRAIN width memory probe; no .pt
    full_run/          initially empty; real training only
```

Read-only project/data paths are supported; output must be writable. Include
any wrapper directory such as Final tier dataset in DATASET_ROOT. PROJECT_ROOT
must contain training/train_crnn.py. The notebook verifies all required paths.

## Dataset packing inspection

Read-only inspection of C:\FYP Dataset Generation\Final_tier_dataset.zip found
1,336,982 entries, compressed file size 1,651,020,735 bytes and uncompressed member
bytes 1,542,480,929 (filesystem overhead is extra). Root directories are train/,
validation/, test/, each with metadata.csv. No wrapper is needed for this ZIP.
The 668,480 samples each have an image AND label, plus auxiliary files: plan for
over 1.3 million filesystem entries/inodes and substantial small-file overhead.

Extracted metadata counts match TRAIN 534,582; validation 66,984; TEST 66,914.
Metadata-derived H=64 width ranges: TRAIN 76–4408, validation 76–3008, TEST 81–3377.
These widths establish individual sample feasibility, not GPU memory capacity.

Use the existing ZIP without repacking or changing images, labels or splits.
Kaggle documents extraction of ZIP uploads for notebook access:
[Kaggle datasets](https://www.kaggle.com/docs/datasets).
Inspect the input mount first. If split directories are exposed, read directly
from /kaggle/input; do not extract or copy again. The current OpenCV dataset reader
requires extracted paths and cannot consume a ZIP filename as DATASET_ROOT.

Only if the mount contains an unextracted ZIP, use this optional manual cell:

```python
from pathlib import Path
import shutil, zipfile
archive = Path('/kaggle/input/your-stage1-dataset/Final_tier_dataset.zip')
destination = Path('/tmp/naqshkash_stage1')
assert not destination.exists(), 'Reuse a complete extraction or choose a fresh destination'
with zipfile.ZipFile(archive) as z:
    needed = sum(member.file_size for member in z.infolist())
    print('Entries:', len(z.infolist()), 'uncompressed bytes:', needed)
    assert shutil.disk_usage('/tmp').free > needed * 2, 'Insufficient disk headroom'
    destination.mkdir()
    z.extractall(destination)  # only your trusted, inspected archive
DATASET_ROOT = destination
```

Check inode/small-file quotas as well as byte capacity. /tmp is ephemeral scratch;
it avoids including 1.3 million extracted data files in saved notebook outputs.
An interrupted extraction must be completed/recreated before training. No local
dataset extraction, duplication or modification was performed for this task.

## Notebook sequence and timing

The notebooks print Python/PyTorch/CUDA/GPU information, assert CUDA, install only
missing non-torch dependencies, verify layout and run actual trainer --help.
Keep the platform's CUDA PyTorch. Missing dependency installation requires Internet.
Then they execute: subset memory probe; 20 training batches saving every 10;
checkpoint inspection; 20 resumed batches; full TRAIN width probe. Expected
counters go epoch=0,next_batch=20,global_step=20 to epoch=0,next_batch=40,global_step=40.
The 4096-sample subset has at least 128 batches at maximum size 32, so no validation
or TEST occurs in the controlled 40-batch test. These notebooks assume the real
dataset, not a tiny fixture with fewer than 128 batches. Use a fresh OUTPUT_ROOT
for a new smoke run. Keep smoke checkpoints separate from full_run.

--preflight-only creates NO checkpoint. last.pt is created by the controlled
training test; best.pt is expected to be absent until validation selects a model.
The first scan still reads ALL TRAIN labels for exact vocabulary and indexes
complete metadata before applying limits. Initial scanning may take minutes with
1.3 million files; GPU computation is short. Width caches do not remove all path
checks. A subset probe does not cover unseen width outliers: the separate full
TRAIN width probe is mandatory before real training. Both probes avoid epoch
training/validation/test. Full-training cells PRINT commands only.

## Exact commands

Run from PROJECT_ROOT; change these input paths once:

```bash
cd /kaggle/input/your-code-dataset/NAQSHKASH
export CUDA_VISIBLE_DEVICES=0
DATASET=/kaggle/input/your-stage1-dataset
OUT=/kaggle/working/naqshkash
```

Quick subset GPU preflight:

```bash
python -m training.train_crnn --dataset-root "$DATASET" --output-root "$OUT/quick_probe" --width-budget 16384 --max-batch-size 32 --amp --num-workers 2 --train-limit 4096 --val-limit 256 --preflight-only
```

Deliberate interruption (20 batches; no epoch validation or TEST):

```bash
python -m training.train_crnn --dataset-root "$DATASET" --output-root "$OUT/resume_test" --width-budget 16384 --max-batch-size 32 --amp --num-workers 2 --epochs 2 --train-limit 4096 --val-limit 256 --skip-test --checkpoint-every-batches 10 --stop-after-batches 20
```

Resume test (20 ADDITIONAL batches per invocation):

```bash
python -m training.train_crnn --dataset-root "$DATASET" --output-root "$OUT/resume_test" --width-budget 16384 --max-batch-size 32 --amp --num-workers 2 --epochs 2 --train-limit 4096 --val-limit 256 --skip-test --checkpoint-every-batches 10 --stop-after-batches 20 --resume "$OUT/resume_test/last.pt"
```

Full TRAIN width coverage before real training (longer first scan):

```bash
python -m training.train_crnn --dataset-root "$DATASET" --output-root "$OUT/full_width_probe" --width-budget 16384 --max-batch-size 32 --amp --num-workers 2 --val-limit 256 --preflight-only
```

Full training — MANUAL ONLY, 30 maximum epochs with early stopping:

```bash
python -m training.train_crnn --dataset-root "$DATASET" --output-root "$OUT/full_run" --width-budget 16384 --max-batch-size 32 --amp --num-workers 2 --epochs 30 --checkpoint-every-batches 1000
```

Normal resume — MANUAL ONLY, restore matching checkpoints first:

```bash
python -m training.train_crnn --dataset-root "$DATASET" --output-root "$OUT/full_run" --width-budget 16384 --max-batch-size 32 --amp --num-workers 2 --epochs 30 --checkpoint-every-batches 1000 --resume "$OUT/full_run/last.pt"
```

Full commands omit subset limits and skip-test: full validation selects by CER;
TEST remains final-only. Smoke checkpoints cannot be promoted by removing limits.

## Memory budget and diagnostics

16384 is an initial UNMEASURED T4 candidate (1,048,576 padded pixels at H=64).
Lower it manually on OOM, e.g. 12288 then 8192, and/or lower maximum batch size,
then reprobe in a fresh smoke folder. Never automatically increase it. A sample
wider than the budget fails explicitly without dropping data; accommodate that
sample, reduce batch size and reprobe. Resume retains saved batching settings.

Trainer prints AMP active, width budget/max batch size, probe geometry and GPU
peak allocated/reserved plus free/total memory in GiB. Probe model/optimizer/RNG
states are restored. Resume logs show zero-based epoch, next_batch and global_step.
Notebook checkpoint inspection prints filenames, sizes, keys, counters and scaler
state. A full probe checks representative high-memory batches, not every possible
future grouping/CUDA workspace condition. Leave headroom.

## Checkpoint persistence

Only last.pt and best.pt are maintained, plus temporary atomic-write files.
Before first validation last.pt suffices; afterward preserve the matching pair,
history and exact settings. Width caches are optional and rebuild if absent.

Kaggle: /kaggle/working is the output location, NOT guaranteed survival across
reset/session expiry. With the run stopped at a checkpoint, download essential
files, or save a notebook version with outputs included and verify its Output.
Saving code alone is not a checkpoint backup. Inspect the current save dialog;
do not launch full training merely to save files. Saved output can be attached
as later input: [Kaggle notebooks](https://www.kaggle.com/docs/notebooks).

To restore a new session, attach saved outputs and copy essential files into a
writable output directory, then run the normal resume command:

```python
from pathlib import Path
import shutil
saved_run = Path('/kaggle/input/your-saved-run/naqshkash/full_run')  # inspect wrapper
destination = OUTPUT_ROOT / 'full_run'
destination.mkdir(parents=True, exist_ok=True)
assert not (destination / 'last.pt').exists(), 'Do not overwrite an active run'
assert (saved_run / 'last.pt').is_file(), 'Saved output has no last.pt'
for name in ('last.pt', 'best.pt', 'training_history.json', 'training_history.csv'):
    source = saved_run / name
    if source.exists():
        shutil.copy2(source, destination / name)
```

After validation best.pt must be alongside last.pt in the output directory:
pointing --resume at a read-only input last.pt alone is insufficient. Preserve a
consistent pair from one stopped run, never mix smoke/full checkpoints.

## Colab backup

Open colab/NAQSHKASH_Stage1_Training.ipynb, select a GPU runtime, and configure
PROJECT_ROOT=/content/NAQSHKASH, DATASET_ROOT=/content/stage1,
OUTPUT_ROOT=/content/naqshkash. Code/data may already be uploaded/downloaded there.
Use the same commands with these paths. Drive is optional, never assumed:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Only if mounting succeeds, set OUTPUT_ROOT under
/content/drive/MyDrive/NAQSHKASH_RUNS. Keep code/data local for small-file speed.
If mounting fails, use /content and download checkpoints regularly via Files
before runtime loss. No OAuth/API integration is needed. /content is ephemeral;
VMs can be deleted after idle/maximum lifetime:
[Colab FAQ](https://research.google.com/colaboratory/faq.html).
Wait for mounted Drive writes/synchronization before disconnecting. Sudden runtime
loss can still lose progress since the last externally preserved checkpoint.

## Common errors

| Error | Action |
|---|---|
| CUDA unavailable / AMP disabled | Enable GPU, restart kernel, keep platform CUDA PyTorch rather than CPU wheels. |
| OOM | Lower budget/max batch size manually; stop competing GPU jobs and reprobe in a fresh smoke output. |
| Dataset path wrong | Point to the directory containing all three splits and metadata.csv, not the ZIP/parent wrapper. |
| ModuleNotFoundError | Set PROJECT_ROOT, run there with python -m training.train_crnn; install only missing dependencies. |
| Incompatible checkpoint | Restore matching code/vocabulary, limits, seed, AMP and batching config; do not bypass checks. Legacy checkpoints are rejected. |
| Missing last.pt | Expected after --preflight-only; run the controlled training test and inspect output/logs. |
| Missing best.pt | Expected before validation; afterward restore the matching best.pt alongside last.pt. |
| Output contains checkpoints | Resume that run or choose a fresh output directory. |

Actual CLI checked using python -m training.train_crnn --help:

```text
--dataset-root --output-root --epochs --batch-size --num-workers --resume
--width-budget --max-batch-size --seed --amp --checkpoint-every-batches
--preflight-only --stop-after-batches --train-limit --val-limit --skip-test
```

There is no --cuda/--device/--gpu option. Trainer auto-selects CUDA; notebook gates
stop before launching commands if no GPU is available.
