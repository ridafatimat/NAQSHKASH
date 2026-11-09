# NAQSHKASH free-GPU notebooks

## Updated Colab workflow

Use `NAQSHKASH_Stage1_Training.ipynb`, running cells in order. Full training and normal
resume are displayed only, including with Run All. Kaggle remains available separately.

1. Runtime → Change runtime type → T4 GPU. Run GPU verification; CUDA must be available.
2. Configure PROJECT_ROOT, DATASET_ROOT, OUTPUT_ROOT and PERSISTENT_CKPT_ROOT near the top.
   Supply PROJECT_ZIP_URL or PROJECT_FILE_ID, and DATASET_ZIP_URL or DATASET_FILE_ID.
   Leave links blank to use already extracted files. Use the updated project ZIP.
3. Install gdown/numpy/opencv-python-headless; keep Colab's preinstalled CUDA torch.
4. Optionally set MOUNT_DRIVE=True and configure a persistent folder under mounted MyDrive.
   Failed credentials fall back to explicit manual backup mode.
5. Download/extract inputs. Nested roots are detected; ambiguous roots fail. Each dataset split
   needs images/, labels/ and metadata.csv, required by the actual loader.
6. Run quick preflight: 256 TRAIN rows, 32 validation rows, sample TRAIN vocabulary only,
   no checkpoint. Tests AMP, CTC backward and Adam allocation on largest padded/longest
   subset batches. This cannot certify unseen wide images.
7. Run controlled 300-batch stop and another 300 using resume. Inspect epoch/next_batch/
   global_step and optimizer/scheduler/scaler state. On T4 the scaler must be nonempty.
   The test uses 32,768 TRAIN samples, 256 validation samples, no TEST, its own output folder.
   Use a clean resume_test folder initially. These are intentional save-and-exit tests.
8. Optionally run the slower full-width probe. Actual full training also performs it at startup.
9. Deliberately execute the displayed full command in a separate cell. All TRAIN and validation
   samples are used. The existing validation CER early stopping (patience 6) may finish before
   the 30-epoch target; best-model selection and final TEST behavior are unchanged.
10. After reset repeat steps 1–5, then restore the full_run backup using RESTORE_CHECKPOINTS=True.
    For manual fallback also set UPLOAD_BACKUP=True to select your downloaded ZIP. Skip the
    smoke test if already verified. Execute the displayed normal resume command.

Resume uses the saved sampler batch offset without loading skipped batches. Full TRAIN
vocabulary/metadata identity checks, width discovery/cache checks and batch planning still run
at startup. Keep data, AMP, width budget, max batch size, seed and limits unchanged. Never
resume a limited smoke checkpoint as a full run. Use distinct persistent folders for Colab
and Kaggle so parallel runs cannot overwrite each other.

### Commands with default paths

Run from the detected project root. The notebook prints exact commands using sys.executable
and the configured detected paths. These shell examples use the default dataset root.
When Drive is unavailable omit --checkpoint-mirror-dir and use manual backup cells.

```bash
# Quick preflight
python -m training.train_crnn --dataset-root /content/NAQSHKASH_DATA --output-root /content/naqshkash_colab/quick_probe --width-budget 16384 --max-batch-size 32 --amp --num-workers 2 --preflight-only --train-limit 256 --val-limit 32

# Deliberate stop
python -m training.train_crnn --dataset-root /content/NAQSHKASH_DATA --output-root /content/naqshkash_colab/resume_test --width-budget 16384 --max-batch-size 32 --amp --num-workers 2 --epochs 30 --train-limit 32768 --val-limit 256 --skip-test --checkpoint-every-batches 100 --stop-after-batches 300 --checkpoint-mirror-dir /content/drive/MyDrive/NAQSHKASH/colab_run_01/resume_test

# Resume test: 300 additional batches
python -m training.train_crnn --dataset-root /content/NAQSHKASH_DATA --output-root /content/naqshkash_colab/resume_test --width-budget 16384 --max-batch-size 32 --amp --num-workers 2 --epochs 30 --train-limit 32768 --val-limit 256 --skip-test --checkpoint-every-batches 100 --stop-after-batches 300 --resume /content/naqshkash_colab/resume_test/last.pt --checkpoint-mirror-dir /content/drive/MyDrive/NAQSHKASH/colab_run_01/resume_test

# Full training: provided only, not executed
python -m training.train_crnn --dataset-root /content/NAQSHKASH_DATA --output-root /content/naqshkash_colab/full_run --width-budget 16384 --max-batch-size 32 --amp --num-workers 2 --epochs 30 --checkpoint-every-batches 1000 --checkpoint-mirror-dir /content/drive/MyDrive/NAQSHKASH/colab_run_01/full_run

# Normal resume after restoring the matching pair
python -m training.train_crnn --dataset-root /content/NAQSHKASH_DATA --output-root /content/naqshkash_colab/full_run --width-budget 16384 --max-batch-size 32 --amp --num-workers 2 --epochs 30 --checkpoint-every-batches 1000 --resume /content/naqshkash_colab/full_run/last.pt --checkpoint-mirror-dir /content/drive/MyDrive/NAQSHKASH/colab_run_01/full_run
```

### Persistence and restore

New optional trainer flag: --checkpoint-mirror-dir PATH. After a successful local atomic save,
copy the completed last.pt or best.pt through a temporary file and replace the persistent file.
After each last.pt save also atomically replace checkpoints.zip with the matching local pair.
Prefer the ZIP for restore: runtime loss between epoch-end best.pt and last.pt saves can leave
individual mirror files from different generations, but the previous ZIP stays consistent.
Before first validation there is no best.pt; last.pt alone is expected.

Mirroring is synchronous, without background polling or OAuth. Failures emit
CHECKPOINT PERSISTENCE FAILED and retain the local run. Resolve failures or back up immediately:
new progress is not durable until persistence succeeds. Allow space for checkpoints, temporary
copies and ZIP. Drive speed and remote sync affect durability; abrupt termination during an
initial remote write cannot be guaranteed safe. Only fixed filenames are retained.

If Drive fails, gdown can download inputs but cannot upload checkpoints. While training is
stopped, set BACKUP_RUN, DOWNLOAD_BACKUP=True and run the backup cell. Retain the downloaded
ZIP on your computer. Repeat at safe stops during a long run. Manual backup is not automatic;
progress after the last completed download can be lost. For automated long-run protection use
working mounted storage. Do not package the active local pair concurrently with training.
During active training you may download the already committed persistent checkpoints.zip.

Restore helper: training.colab_checkpoints.restore(archive, output). It verifies ZIP names and
CRCs and restores last.pt plus optional best.pt into a folder without .pt files. Only restore
trusted checkpoints. After completed validation the trainer requires both files. The restore
cell handles mounted storage or manual upload. No upload happens automatically.

Free Colab may disconnect unpredictably. Browser disconnect and runtime termination differ.
An open browser does not guarantee uninterrupted runtime. Mid-epoch checkpoints limit lost
work only to the last successfully persisted save. /content is ephemeral.

### Common errors

| Error | Action |
|---|---|
| ModuleNotFoundError | Check detected PROJECT_ROOT, change into it, install notebook dependencies and use the updated project ZIP. |
| CUDA unavailable | Enable T4 GPU runtime; retain CUDA torch instead of installing CPU wheels. |
| OOM | Lower width budget/max batch size and rerun preflight. 16384 is a starting point. Resume requires the saved batch plan; a new budget requires a separate new run. A single sample wider than budget needs a sufficient budget and GPU memory; do not drop it. |
| Missing last.pt | Preflight creates none. Restore a completed backup for the correct run; without one lost progress is unrecoverable. |
| Drive credentials | Use gdown for inputs and manual ZIP download/upload for checkpoints. Input downloads do not persist outputs. |
| Wrong dataset path | Root must contain train/validation/test, each with images/labels/metadata.csv. Check the detected nested root. |
| Resume mismatch | Restore the matching ZIP and use unchanged data, AMP, seed, width budget, max batch size and limits. |
| gdown access/quota | Check share access/file ID/quota, or use already extracted inputs. |

---

## Existing Kaggle workflow

Kaggle: attach code and extracted Stage 1 inputs; output under /kaggle/working.
Its notebook retains its existing 20 + 20 controlled-batch test.
See [GPU deployment](../training/GPU_DEPLOYMENT.md) and [trainer details](../training/PORTABLE_TRAINING.md).
