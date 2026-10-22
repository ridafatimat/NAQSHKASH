# Inspection before implementation

Inspected C:/NAQSHKASH on October 18, 2026 before changing training code:
training/train_crnn.py, gpu_deployment.py, colab_checkpoints.py; data/tier_dataset.py,
collate.py, width_sampler.py, vocabulary.py, encoder.py, ctc_decode.py; models/crnn.py;
tests/test_training_portable.py and test_gpu_deployment.py; existing Colab/Kaggle
notebook workflow and README. Approved stratified manifest header/rows and existing
validation_report.json were also inspected. No AGENTS.md was found in the project.

| Area | Existing behavior | Manifest training path |
|---|---|---|
| Dataset | Reads split metadata.csv, resolves and checks every image and label | Reads only chosen CSV manifests; checks split, labels/widths/feasibility in memory |
| Labels | Reopens TXT for every fetched sample; full TRAIN scan builds vocabulary | Uses training_label directly; fixed original full-TRAIN 65-symbol mapping |
| Widths | Stats every image for cache identity, opens PNG headers on cache miss | Uses resized_width; validates dimensions when pixels are fetched |
| Startup | Full labels + metadata/path inventory + widths + worst-batch forward/backward/Adam probes | Manifest hashes/parse and first 100 pixel checks per split; no full image inventory/probe |
| Batches | Seed+epoch shuffle, pool sort, padded width budget, max batch cap; direct start_batch resume | Retains same sampler and algorithm; uses manifest widths |
| Normalization | Grayscale /255; bilinear height 64, aspect ratio; white right padding | Unchanged existing collate |
| AMP/CTC | Forward autocast, logits.float().log_softmax, zero_infinity=True | Preserved; CPU lengths avoid extra tiny GPU transfers |
| Clipping | scale/backward, unscale, norm clip 5, step/update | Preserved order; finite-gradient guard skips failed update |
| Nonfinite | Raises on any nonfinite loss, duplicate finite checks | Warns/skips, aborts at cumulative threshold; scheduler advances only on update |
| Scheduler | ReduceLROnPlateau after epoch validation | Linear warmup then cosine floor 0.001 of initial LR, every optimizer step |
| Logging | loss.item every batch; prints first/every 1000 train, first/every 500 validation | Detached loss accumulator; reports every 100 optimizer steps by default, append CSV |
| Validation | Entire validation split each epoch, greedy, CER/exact overall/per tier | Approved 10k manifest, configurable step cadence; same decoder/metrics |
| Test | Automatically evaluates test at final end unless skip-test | No test access; separate final evaluation |
| Checkpoints | Atomic replace v2 model/optimizer/scheduler/scaler/progress/Python/Torch/CUDA RNG, optional mirror; no NumPy/hash | Atomic fsynced v3, NumPy RNG, vocabulary/manifest hashes, compatible config and sampler checks |
| Resume | v2 map/class/width/config checks, direct batch offset, requires matching best file after validation | v3 exact schedule/identity/AMP checks, restore RNG, direct offset, auto or input path |
| Loader | num-workers configurable; training pin_memory follows CUDA; budget loader did not enable persistent_workers/prefetch controls | workers 4, pin/persistent true, prefetch 2; all configurable; worker-zero handling |
| Runtime | No runtime guard; epoch target default 30 | Explicit total steps or benchmark; 660-minute default, 120-second shutdown margin |

Existing CRNN is unchanged (hidden size 256, two BiLSTM layers, 66 classes).
Legacy CLI remains to satisfy existing users/tests; select the optimized path with
--total-steps, --benchmark-steps or explicit manifest arguments. Old notebooks and
gpu_deployment.commands() describe the legacy epoch workflow, so use KAGGLE_STAGE1.md
for this phase. The new CLI is the thin launch interface; YAML is unnecessary.
