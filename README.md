# NAQSHKASH

NAQSHKASH recognizes Kashmiri carpet Talim tiers using a CNN, a two-layer
bidirectional LSTM (hidden size 256), and CTC. Input is grayscale at height 64,
with aspect ratio preserved and white right padding. There are 65 target
characters and blank index 0 (66 output classes).

The Stage-1 pipeline is integrated into this repository root alongside Ayesha's
row preprocessing and CTC APIs. The original Talamat rule references are vendored
under `talim/reference/`. Run the commands below from the repository root.

## Layout

| Directory | Purpose |
| --- | --- |
| `talim/` | Talamat-derived notation rules and validation |
| `preprocessing/` | Cleaning, deskew, margin crop, ruling lines and logical rows |
| `data/` | Page/row/tier readers, builders, manifests, stratification, CTC encoding |
| `models/` | CRNN architecture |
| `training/` | Legacy epoch trainer, manifest step trainer, resume and GPU helpers |
| `evaluation/` | Saved-checkpoint evaluation command |
| `results/stage1/` | Original final evaluation, training metrics and failure logs |
| `.github/workflows/` | Combined root tests on GitHub Actions |
| `config/`, `tests/`, `colab/` | Fixed vocabulary, tests and older GPU workflows |

## Existing preprocessing functionality


The `preprocessing` package provides an end-to-end preprocessing pipeline for Talim documents:

```text
preprocessing/
├── __init__.py        # Unified package exports
├── deskew.py          # Rotational deskewing and skew angle estimation
├── crop.py            # Margin detection and document content bounding-box cropping
├── rows.py            # 3-tier logical row detection and separation
└── pipeline.py        # Integrated end-to-end preprocessing pipeline
```

### Preprocessing Capabilities:
1. **Rotational Deskewing (`preprocessing.deskew`)**: Corrects small global in-plane rotation using projection-profile analysis with confidence gating.
2. **Margin Cropping (`preprocessing.crop`)**: Removes outer blank borders and dark scanner margins while preserving text strokes.
3. **Logical Row Detection & Separation (`preprocessing.rows`)**: Extracts complete 3-tier Talim logical rows (Upper symbols, Center stitch counts, Lower symbols) as single units.
4. **End-to-End Pipeline (`preprocessing.pipeline`)**: Seamlessly coordinates deskewing, margin cropping, row detection, row extraction, and local-to-global coordinate mapping into structured `PreprocessingResult` dataclasses.

For full details on Ayesha's cropping & row detection design, see [README_AYESHA.md](README_AYESHA.md).


## Vocabulary compatibility

Ayesha's `TalimVocabulary()` keeps its 67-class multi-line row alphabet, caller
ordering, serialization and canonical label-file fallback. Stage-1 uses
`TalimVocabulary.for_tiers(characters)` for sorted explicit tier alphabets and
`build_tier_vocabulary_from_labels(path)` for strict TRAIN-only discovery. The
fixed mapping in `config/vocabulary.json` has 65 characters plus blank (66 classes).
The existing encoder, structured CTC parser, tolerant `decode_path`, normalization
and model-preparation APIs remain available. `greedy_ctc_decode` provides strict
tier decoding using the same CTC-collapse algorithm.

## Setup and data

Use Python 3.10+ and install `requirements.txt` in a separate environment.
For GPU training retain the platform's compatible CUDA-enabled PyTorch rather
than replacing it with a CPU build. Commands below run from the repository root:

```bash
python -m pip install -r requirements.txt
python -m pytest tests -q
```

The companion FYP-Dataset_Generation repository renders Talamat glyphs into PNG pages with TXT/JSON
labels and split metadata. From that companion repository root its command is
`python dataset_tools/generate_dataset.py`; see
[generator documentation](https://github.com/ridafatimat/FYP-Dataset_Generation/blob/main/dataset_tools/README.md) for configuration and seeds.
The current generator configuration alone does not establish that a newly
generated dataset is byte-identical to the historical 100k training corpus.

The full tier builder reads the historical ZIP layout
`datasets/talim_100000/{train,validation,test}`. It preserves page split membership,
crops logical rows, aligns them with JSON ground truth, and separates nonempty
upper/count/lower tiers. Full-builder deskew is disabled for clean synthetic pages;
the general preprocessing pipeline also supports deskew and real scan cleanup.

```bash
python -m data.build_full_tier_dataset --zip-path /path/talim_100000.zip --output-base /path/tiers --split train
python -m data.build_full_tier_dataset --zip-path /path/talim_100000.zip --output-base /path/tiers --split validation
python -m data.build_full_tier_dataset --zip-path /path/talim_100000.zip --output-base /path/tiers --split test
python -m data.build_training_manifest --dataset-root /path/tiers
python -m data.build_stratified_subset --manifest /path/tiers/manifests/full_tier_manifest.csv --dataset-root /path/tiers
```

The manifest records relative image paths, raw and training labels, widths,
parent IDs and CTC feasibility. Leading ASCII spaces become layout offsets;
internal spaces remain recognition targets. Stratification selects 150,000
TRAIN and 10,000 fast-validation tiers and checks parent/tier leakage. The
manifest trainer reads approved CSVs, checks vocabulary and CTC feasibility,
and verifies manifest hashes when resuming. Full validation and test are reserved
for separate final evaluation. Datasets and generated manifests are external
artifacts; this code export alone cannot recreate the exact historical split
without the original dataset and approved manifests.

## Training and evaluation

The saved best checkpoint records these production settings (GPU required for
matching AMP behavior; worker throughput varies by machine):

```bash
python -m training.train_crnn --dataset-root /path/tiers --output-root /path/run --total-steps 16000 --warmup-steps 1000 --amp-init-scale 1024 --width-budget 32768 --max-batch-size 64 --num-workers 4 --checkpoint-every-steps 2000 --validate-every-steps 2000 --max-runtime-minutes 90
```

Resume with the same schedule and `--resume auto` in the same output directory.
The trainer saves optimizer, scheduler, scaler, sampler and RNG state. Nonfinite
batches skip the optimizer and scheduler and produce diagnostic logs. The
archived run includes two nonfinite-gradient events; original logs are retained.
See [manifest training guide](training/KAGGLE_STAGE1.md) for benchmarking and resume.
The two `colab/` notebooks exercise the older epoch workflow and are retained
as working compatibility examples; they do not reproduce the final 16k manifest run.

```bash
python -m evaluation.evaluate_checkpoint --checkpoint /path/best.pt --dataset-root /path/tiers --output /path/new_evaluation.json
```

This command loads saved weights and vocabulary, performs greedy CTC decoding,
and reports CER (total character edits / total reference characters), exact
sequence accuracy and per-tier metrics over every requested split sample.
It is also the batch inference path; application integration remains future work.

## Verified saved results

| Held-out synthetic split | Tiers | CER | Exact sequence accuracy |
| --- | ---: | ---: | ---: |
| Full validation | 66,984 | 0.1267% | 99.4058% |
| Untouched test | 66,914 | 0.1057% | 99.4740% |

These are **held-out synthetic-tier results, not real-handwriting accuracy**.
They were verified from saved JSON/CSV, not recomputed on the unavailable full
dataset during this export. Original filenames and bytes are preserved in
[results/stage1](results/stage1/). CER and accuracy in JSON/CSV are fractions.
The JSON retains its original Kaggle checkpoint path as provenance.

`artifact_integrity.json` records sizes and SHA-256 hashes for all checkpoint ZIP
entries. Both archives pass ZIP CRC integrity checks. `best.pt` is 92,374,435
bytes (88.10 MiB); it is excluded from Git, as are `last.pt`, datasets and ZIPs.
The prepared release tag is `stage1-crnn-ctc-v1.0.0`. Download availability will be
listed on the repository's [Releases page](https://github.com/ridafatimat/NAQSHKASH/releases)
after the owner approves publication; no model release is currently claimed.

Stage 1 synthetic OCR training and saved final evaluation are complete.
Real-handwriting adaptation, Transformer comparison and application integration
remain future work. Historical optimization notes record earlier inspections;
their local paths and old execution reports are provenance, not installation
requirements or validation of this export. Tests requiring external sample
datasets are explicitly skipped when those datasets are absent.

## Preprocessing validation and benchmark

The existing deskew validation and preprocessing benchmark remain available:

```bash
python tests/test_deskew.py
python run_ayesha_preprocessing.py
```

Their image fixtures and generated overlays are external artifacts. See
[README_AYESHA.md](README_AYESHA.md) for normalization, model preparation and
multi-line CTC label parsing examples.
