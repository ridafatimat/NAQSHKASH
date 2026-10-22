"""
NAQSHKASH
Production CRNN + CTC training for Stage 1 clean synthetic data.

Dataset:
    <dataset-root>/train, validation, test

Splits:
    train
    validation
    test

Important:
- Vocabulary is built from TRAIN only.
- Validation is used for model selection.
- Test is evaluated only after training is finished.
- Best model is selected using validation CER.
- Checkpoints contain model/optimizer/scheduler state and training config.
"""

from __future__ import annotations

import argparse
from training.colab_checkpoints import mirror_checkpoint, quick_dataset
import os
from data.width_sampler import WidthBudgetBatchSampler, resized_widths
import csv
import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from data.tier_dataset import TalimTierDataset
from data.collate import talim_collate_fn
from data.vocabulary import TalimVocabulary
from data.encoder import CTCLabelEncoder
from data.ctc_decode import greedy_ctc_decode
from models.crnn import CRNN


# ============================================================
# CONFIG
# ============================================================

CHECKPOINT_MIRROR_DIR = None
IMAGE_HEIGHT = 64

HIDDEN_SIZE = 256
NUM_LSTM_LAYERS = 2

DEFAULT_EPOCHS = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

GRAD_CLIP = 5.0

RANDOM_SEED = 42

# Scheduler
LR_FACTOR = 0.5
LR_PATIENCE = 2
MIN_LR = 1e-6

# Early stopping uses VALIDATION CER only.
EARLY_STOP_PATIENCE = 6

# Windows-safe default.
NUM_WORKERS = 0


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int):

    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# DEVICE
# ============================================================

def get_device():

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# ============================================================
# EDIT DISTANCE / CER
# ============================================================

def levenshtein_distance(
    reference: str,
    prediction: str,
) -> int:
    """
    Character-level edit distance.
    """

    if reference == prediction:
        return 0

    if not reference:
        return len(prediction)

    if not prediction:
        return len(reference)

    previous = list(
        range(
            len(prediction) + 1
        )
    )

    for i, ref_char in enumerate(
        reference,
        start=1,
    ):

        current = [i]

        for j, pred_char in enumerate(
            prediction,
            start=1,
        ):

            insertion = (
                current[j - 1] + 1
            )

            deletion = (
                previous[j] + 1
            )

            substitution = (
                previous[j - 1]
                + (
                    0
                    if ref_char == pred_char
                    else 1
                )
            )

            current.append(
                min(
                    insertion,
                    deletion,
                    substitution,
                )
            )

        previous = current

    return previous[-1]


# ============================================================
# DECODING
# ============================================================

def decode_predictions(
    logits,
    input_lengths,
    vocabulary,
):
    """
    Greedy CTC decode.

    logits:
        [T, B, C]
    """

    predictions = torch.argmax(
        logits,
        dim=2,
    )

    # [B, T]
    predictions = predictions.permute(
        1,
        0,
    )

    decoded = []

    for batch_index in range(
        predictions.shape[0]
    ):

        length = int(
            input_lengths[
                batch_index
            ].item()
        )

        indices = (
            predictions[
                batch_index,
                :length,
            ]
            .detach()
            .cpu()
            .tolist()
        )

        text = greedy_ctc_decode(
            indices,
            vocabulary,
        )

        decoded.append(text)

    return decoded


# ============================================================
# VOCABULARY SERIALIZATION
# ============================================================

def extract_train_characters(
    labels_root: Path,
):
    """
    Store the actual TRAIN character set in the checkpoint.

    Important:
    Only leading alignment spaces are removed,
    matching TalimTierDataset OCR targets.

    Internal spaces remain legitimate characters.
    """

    characters = set()

    label_files = sorted(
        labels_root.glob("*.txt")
    )

    for path in label_files:

        text = (
            path.read_text(
                encoding="utf-8"
            )
            .rstrip("\r\n")
            .lstrip(" ")
        )

        characters.update(text)

    return sorted(characters)


# ============================================================
# METRICS
# ============================================================

def empty_tier_metrics():

    return {
        "samples": 0,
        "exact": 0,
        "edit_distance": 0,
        "characters": 0,
    }


def finalize_tier_metrics(data):

    result = {}

    for tier_name, values in data.items():

        samples = values["samples"]

        exact_accuracy = (
            values["exact"]
            / samples
            if samples
            else 0.0
        )

        cer = (
            values["edit_distance"]
            / values["characters"]
            if values["characters"]
            else 0.0
        )

        result[tier_name] = {
            "samples":
                samples,

            "exact_accuracy":
                exact_accuracy,

            "cer":
                cer,
        }

    return result


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    encoder,
    criterion,
    optimizer,
    device,
    scaler=None, progress=None, on_batch=None,
):

    model.train()

    progress = progress if progress is not None else {}
    total_loss = progress.get("loss_sum", 0.0)
    total_batches = progress.get("loss_batches", 0)
    scaler = scaler or torch.amp.GradScaler("cuda", enabled=False)

    start_time = time.time()

    for batch_index, batch in enumerate(
        loader,
        start=1,
    ):

        images = (
            batch["images"]
            .to(
                device,
                non_blocking=True,
            )
        )

        image_widths = (
            batch["image_widths"]
            .to(
                device,
                non_blocking=True,
            )
        )

        label_texts = batch[
            "label_texts"
        ]

        encoded = encoder.encode_batch(
            label_texts
        )

        targets = (
            encoded["targets"]
            .to(
                device,
                non_blocking=True,
            )
        )

        target_lengths = (
            encoded[
                "target_lengths"
            ]
            .to(
                device,
                non_blocking=True,
            )
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
            logits = model(images)

        log_probs = torch.log_softmax(
            logits.float(),
            dim=2,
        )

        input_lengths = (
            model.output_lengths(
                image_widths
            )
        )

        if torch.any(
            target_lengths
            > input_lengths
        ):

            raise RuntimeError(
                "CTC target length exceeds "
                "available CRNN sequence length."
            )

        loss = criterion(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
        )

        if not torch.isfinite(loss):

            raise RuntimeError(
                "Non-finite training loss encountered."
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=GRAD_CLIP,
        )

        scaler.step(optimizer)
        scaler.update()

        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite CTC loss encountered.")

        total_loss += float(
            loss.item()
        )

        total_batches += 1
        progress.update(loss_sum=total_loss, loss_batches=total_batches,
                        next_batch=progress.get("next_batch", 0) + 1,
                        global_step=progress.get("global_step", 0) + 1)
        if on_batch:
            on_batch(progress)

        if (
            batch_index == 1
            or batch_index % 1000 == 0
        ):

            elapsed = (
                time.time()
                - start_time
            )

            print(
                f"    train batch "
                f"{batch_index:,}/"
                f"{len(loader):,} | "
                f"loss={loss.item():.4f} | "
                f"time={elapsed / 60:.1f} min"
            )

    average_loss = (
        total_loss
        / max(
            1,
            total_batches,
        )
    )

    return average_loss


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    encoder,
    criterion,
    vocabulary,
    device,
    split_name,
    validate_ctc_runtime=False,
):

    model.eval()

    total_loss = 0.0
    total_batches = 0

    total_samples = 0
    exact_matches = 0

    total_edits = 0
    total_characters = 0

    tier_metrics = {
        "upper":
            empty_tier_metrics(),

        "count":
            empty_tier_metrics(),

        "lower":
            empty_tier_metrics(),
    }

    examples = []

    start_time = time.time()

    for batch_index, batch in enumerate(
        loader,
        start=1,
    ):

        images = (
            batch["images"]
            .to(
                device,
                non_blocking=True,
            )
        )

        image_widths = (
            batch["image_widths"]
            .to(
                device,
                non_blocking=True,
            )
        )

        truths = batch[
            "label_texts"
        ]

        tier_names = batch.get(
            "tier_names",
            [
                "unknown"
                for _ in truths
            ],
        )

        sample_ids = batch.get(
            "sample_ids",
            [
                ""
                for _ in truths
            ],
        )

        encoded = encoder.encode_batch(
            truths
        )

        targets = (
            encoded["targets"]
            .to(
                device,
                non_blocking=True,
            )
        )

        target_lengths = (
            encoded[
                "target_lengths"
            ]
            .to(
                device,
                non_blocking=True,
            )
        )

        logits = model(
            images
        )

        log_probs = torch.log_softmax(
            logits.float(),
            dim=2,
        )

        input_lengths = (
            model.output_lengths(
                image_widths
            )
        )

        if validate_ctc_runtime:
            from training.manifest_training import runtime_ctc_lengths, fp32_ctc_loss
            input_lengths, _ = runtime_ctc_lengths(model, batch, encoded, logits.shape[0])
            loss, log_probs = fp32_ctc_loss(logits, targets, input_lengths, encoded['target_lengths'], criterion)
        else:
            loss = criterion(
                log_probs,
                targets,
                input_lengths,
                target_lengths,
            )

        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite CTC loss encountered.")

        total_loss += float(
            loss.item()
        )

        total_batches += 1

        predictions = decode_predictions(
            logits,
            input_lengths,
            vocabulary,
        )

        for (
            sample_id,
            truth,
            prediction,
            tier_name,
        ) in zip(
            sample_ids,
            truths,
            predictions,
            tier_names,
        ):

            total_samples += 1

            is_exact = (
                truth == prediction
            )

            if is_exact:
                exact_matches += 1

            edit_distance = (
                levenshtein_distance(
                    truth,
                    prediction,
                )
            )

            total_edits += (
                edit_distance
            )

            total_characters += len(
                truth
            )

            if tier_name in tier_metrics:

                values = tier_metrics[
                    tier_name
                ]

                values["samples"] += 1

                values["exact"] += int(
                    is_exact
                )

                values[
                    "edit_distance"
                ] += edit_distance

                values[
                    "characters"
                ] += len(
                    truth
                )

            # Keep a few wrong predictions
            if (
                not is_exact
                and len(examples) < 20
            ):

                examples.append(
                    {
                        "sample_id":
                            sample_id,

                        "tier_name":
                            tier_name,

                        "truth":
                            truth,

                        "prediction":
                            prediction,

                        "edit_distance":
                            edit_distance,
                    }
                )

        if (
            batch_index == 1
            or batch_index % 500 == 0
        ):

            elapsed = (
                time.time()
                - start_time
            )

            print(
                f"    {split_name} batch "
                f"{batch_index:,}/"
                f"{len(loader):,} | "
                f"time={elapsed / 60:.1f} min"
            )

    average_loss = (
        total_loss
        / max(
            1,
            total_batches,
        )
    )

    exact_accuracy = (
        exact_matches
        / total_samples
        if total_samples
        else 0.0
    )

    cer = (
        total_edits
        / total_characters
        if total_characters
        else 0.0
    )

    return {
        "loss":
            average_loss,

        "samples":
            total_samples,

        "exact_matches":
            exact_matches,

        "exact_accuracy":
            exact_accuracy,

        "edit_distance":
            total_edits,

        "characters":
            total_characters,

        "cer":
            cer,

        "per_tier":
            finalize_tier_metrics(
                tier_metrics
            ),

        "error_examples":
            examples,
    }


# ============================================================
# CHECKPOINT
# ============================================================

def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    epoch,
    best_val_cer,
    vocabulary,
    train_characters,
    history,
    config,
    scaler=None, progress=None, sampler_state=None,
):

    checkpoint = {
        "format_version": 2,
        "next_batch": (progress or {}).get("next_batch", 0),
        "global_step": (progress or {}).get("global_step", 0),
        "progress": dict(progress or {}),
        "sampler_state": sampler_state,
        "scaler_state_dict": scaler.state_dict() if scaler else {},
        "char_to_idx": dict(vocabulary.char_to_index),
        "idx_to_char": dict(vocabulary.index_to_char),
        "vocabulary": {"characters": vocabulary.characters,
                       "blank_token": vocabulary.blank_token,
                       "blank_index": vocabulary.blank_index},
        "random_seed": config.get("random_seed", RANDOM_SEED),
        "optimizer_settings": {"name": "AdamW", "lr": LEARNING_RATE,
                               "weight_decay": WEIGHT_DECAY},
        "epochs_without_improvement": config["epochs_without_improvement"],
        "random_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "epoch":
            epoch,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "scheduler_state_dict":
            scheduler.state_dict(),

        "best_val_cer":
            best_val_cer,

        "blank_index":
            vocabulary.blank_index,

        "num_classes":
            len(vocabulary),

        # Explicit actual training character set.
        "train_characters":
            train_characters,

        "image_height":
            IMAGE_HEIGHT,

        "architecture": {
            "name":
                "CRNN",

            "hidden_size":
                HIDDEN_SIZE,

            "num_lstm_layers":
                NUM_LSTM_LAYERS,
        },

        "dataset": {
            "root":
                str(DATASET_ROOT),

            "train":
                str(TRAIN_PATH),

            "validation":
                str(VALIDATION_PATH),

            "test":
                str(TEST_PATH),
        },

        "config":
            config,

        "history":
            history,
    }

    # Replace only after a complete write; retain the previous checkpoint on failure.
    temporary = Path(str(path) + ".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)
    mirror_checkpoint(path, CHECKPOINT_MIRROR_DIR)


# ============================================================
# HISTORY
# ============================================================

def save_history(
    history,
):

    HISTORY_JSON.write_text(
        json.dumps(
            history,
            indent=2,
        ),
        encoding="utf-8",
    )

    fieldnames = [
        "epoch",
        "learning_rate",
        "train_loss",
        "val_loss",
        "val_cer",
        "val_exact_accuracy",
        "upper_cer",
        "upper_exact_accuracy",
        "count_cer",
        "count_exact_accuracy",
        "lower_cer",
        "lower_exact_accuracy",
    ]

    with HISTORY_CSV.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for item in history:

            per_tier = item[
                "validation"
            ][
                "per_tier"
            ]

            writer.writerow(
                {
                    "epoch":
                        item["epoch"],

                    "learning_rate":
                        item[
                            "learning_rate"
                        ],

                    "train_loss":
                        item[
                            "train_loss"
                        ],

                    "val_loss":
                        item[
                            "validation"
                        ][
                            "loss"
                        ],

                    "val_cer":
                        item[
                            "validation"
                        ][
                            "cer"
                        ],

                    "val_exact_accuracy":
                        item[
                            "validation"
                        ][
                            "exact_accuracy"
                        ],

                    "upper_cer":
                        per_tier[
                            "upper"
                        ][
                            "cer"
                        ],

                    "upper_exact_accuracy":
                        per_tier[
                            "upper"
                        ][
                            "exact_accuracy"
                        ],

                    "count_cer":
                        per_tier[
                            "count"
                        ][
                            "cer"
                        ],

                    "count_exact_accuracy":
                        per_tier[
                            "count"
                        ][
                            "exact_accuracy"
                        ],

                    "lower_cer":
                        per_tier[
                            "lower"
                        ][
                            "cer"
                        ],

                    "lower_exact_accuracy":
                        per_tier[
                            "lower"
                        ][
                            "exact_accuracy"
                        ],
                }
            )


# ============================================================
# PRINT METRICS
# ============================================================

def print_evaluation(
    name,
    metrics,
):

    print(
        f"{name} loss: "
        f"{metrics['loss']:.4f}"
    )

    print(
        f"{name} CER: "
        f"{metrics['cer']:.4f} "
        f"({metrics['cer'] * 100:.2f}%)"
    )

    print(
        f"{name} exact: "
        f"{metrics['exact_accuracy']:.4f} "
        f"({metrics['exact_accuracy'] * 100:.2f}%)"
    )

    print(
        f"{name} exact matches: "
        f"{metrics['exact_matches']:,}/"
        f"{metrics['samples']:,}"
    )

    for tier_name in (
        "upper",
        "count",
        "lower",
    ):

        values = metrics[
            "per_tier"
        ][
            tier_name
        ]

        print(
            f"  {tier_name:<5} | "
            f"CER "
            f"{values['cer'] * 100:6.2f}% | "
            f"Exact "
            f"{values['exact_accuracy'] * 100:6.2f}% | "
            f"N={values['samples']:,}"
        )


# ============================================================
# MAIN
# ============================================================

def limited_dataset(path, limit):
    dataset = TalimTierDataset(path)
    if limit is not None:
        return Subset(dataset, range(min(limit, len(dataset))))
    return dataset


def make_loader(dataset, batch_size, workers, device, shuffle=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers, pin_memory=device.type == "cuda",
                      persistent_workers=workers > 0, collate_fn=talim_collate_fn)


def main():
    parser = argparse.ArgumentParser(description="Stage 1 clean synthetic CRNN training")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                        help="Total epoch target, including completed resume epochs")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--checkpoint-mirror-dir", type=Path, help="Persistent folder; synchronously mirror completed saves and a consistent restore ZIP")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--width-budget", type=int, default=16384)
    parser.add_argument("--max-batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--checkpoint-every-batches", type=int, default=1000)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--stop-after-batches", type=int,
                        help="Save and exit after this many batches in this invocation; no validation/test")
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--val-limit", type=int)
    parser.add_argument("--skip-test", action="store_true",
                        help="Preflight: do not load or evaluate TEST")
    args = parser.parse_args()
    global CHECKPOINT_MIRROR_DIR
    CHECKPOINT_MIRROR_DIR = args.checkpoint_mirror_dir
    if CHECKPOINT_MIRROR_DIR and CHECKPOINT_MIRROR_DIR.resolve() == args.output_root.resolve():
        parser.error("Checkpoint mirror must differ from output-root")
    if args.epochs < 1 or args.num_workers < 0:
        parser.error("epochs must be positive and num-workers nonnegative")
    for name in ("batch_size", "max_batch_size", "width_budget", "checkpoint_every_batches", "stop_after_batches", "train_limit", "val_limit"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(name + " must be positive")

    global DATASET_ROOT, TRAIN_PATH, VALIDATION_PATH, TEST_PATH, OUTPUT_ROOT
    global HISTORY_CSV, HISTORY_JSON, BEST_CHECKPOINT, LAST_CHECKPOINT, TEST_REPORT
    DATASET_ROOT = args.dataset_root.resolve()
    TRAIN_PATH, VALIDATION_PATH, TEST_PATH = [DATASET_ROOT / x for x in ("train", "validation", "test")]
    OUTPUT_ROOT = args.output_root.resolve()
    HISTORY_CSV = OUTPUT_ROOT / "training_history.csv"
    HISTORY_JSON = OUTPUT_ROOT / "training_history.json"
    BEST_CHECKPOINT = OUTPUT_ROOT / "best.pt"
    LAST_CHECKPOINT = OUTPUT_ROOT / "last.pt"
    TEST_REPORT = OUTPUT_ROOT / "test_results.json"
    for split in (TRAIN_PATH, VALIDATION_PATH, TEST_PATH):
        if not split.is_dir():
            parser.error(f"Missing dataset split: {split}")
    checkpoint = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        if checkpoint.get("format_version") != 2:
            parser.error("Legacy epoch checkpoint cannot resume with the new sampler; start a new output run")
    elif any(OUTPUT_ROOT.glob("*.pt")):
        parser.error("Output already contains checkpoints; use --resume or a new output folder")
    set_seed(args.seed)
    device = get_device()
    batch_size = args.batch_size or (32 if device.type == "cuda" else 8)
    print(f"Device: {device}; batch size: {batch_size}; workers: {args.num_workers}", flush=True)
    # All saving runs use full TRAIN vocabulary; bounded memory-only probes do not save.
    quick = args.preflight_only and args.train_limit is not None and args.resume is None
    if quick:
        train_dataset = quick_dataset(TRAIN_PATH, args.train_limit)
        train_characters = sorted({c for sample in train_dataset.samples
            for c in sample['label_path'].read_text(encoding='utf-8').rstrip('\r\n').lstrip(' ')})
        print("Bounded preflight: sample vocabulary only; no checkpoint; full-run startup probes all TRAIN widths", flush=True)
    else:
        train_characters = extract_train_characters(TRAIN_PATH / "labels")
    if not train_characters:
        raise ValueError("TRAIN labels contain no OCR characters")
    vocabulary = TalimVocabulary(train_characters)
    if checkpoint and (checkpoint["char_to_idx"] != vocabulary.char_to_index or
                       checkpoint["idx_to_char"] != vocabulary.index_to_char or
                       checkpoint["blank_index"] != vocabulary.blank_index):
        raise ValueError("Resume vocabulary differs from current TRAIN vocabulary")
    encoder = CTCLabelEncoder(vocabulary)
    if not quick:
        train_dataset = limited_dataset(TRAIN_PATH, args.train_limit)
    val_dataset = (quick_dataset(VALIDATION_PATH, args.val_limit or 32) if quick
                   else limited_dataset(VALIDATION_PATH, args.val_limit))
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    max_batch_size = args.max_batch_size or args.batch_size or 32
    widths, fingerprint = resized_widths(train_dataset, OUTPUT_ROOT / "train_widths.json")
    sampler = WidthBudgetBatchSampler(widths, args.width_budget, max_batch_size, args.seed)
    sampler_state = dict(version=sampler.VERSION, fingerprint=fingerprint,
                         widths=widths, pool_size=sampler.pool_size)
    def budget_loader(dataset, batch_sampler):
        return DataLoader(dataset, batch_sampler=batch_sampler, num_workers=args.num_workers,
                          pin_memory=device.type == "cuda", collate_fn=talim_collate_fn,
                          generator=torch.Generator().manual_seed(args.seed))
    train_loader = budget_loader(train_dataset, sampler)
    val_widths, _ = resized_widths(val_dataset, OUTPUT_ROOT / "validation_widths.json")
    val_loader = budget_loader(val_dataset, WidthBudgetBatchSampler(val_widths, args.width_budget, max_batch_size, args.seed))
    print(f"TRAIN: {len(train_dataset):,}; validation: {len(val_dataset):,}; classes: {len(vocabulary)}", flush=True)
    model = CRNN(len(vocabulary), hidden_size=HIDDEN_SIZE, num_lstm_layers=NUM_LSTM_LAYERS).to(device)
    criterion = nn.CTCLoss(blank=vocabulary.blank_index, reduction="mean", zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=LR_FACTOR, patience=LR_PATIENCE, min_lr=MIN_LR)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    print(f"AMP active: {scaler.is_enabled()} | width budget: {args.width_budget:,} | "
          f"max batch size: {max_batch_size}", flush=True)
    if args.amp and not scaler.is_enabled():
        print("CUDA unavailable: AMP disabled; using float32", flush=True)
    config = dict(width_budget=args.width_budget, max_batch_size=max_batch_size, amp=args.amp, stage="clean_synthetic", epochs=args.epochs, batch_size=batch_size,
                  num_workers=args.num_workers, train_limit=args.train_limit, val_limit=args.val_limit,
                  random_seed=args.seed, learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
                  gradient_clip=GRAD_CLIP, image_height=IMAGE_HEIGHT, hidden_size=HIDDEN_SIZE,
                  num_lstm_layers=NUM_LSTM_LAYERS, early_stop_patience=EARLY_STOP_PATIENCE,
                  scheduler="ReduceLROnPlateau", scheduler_factor=LR_FACTOR,
                  scheduler_patience=LR_PATIENCE, min_lr=MIN_LR, best_model_metric="validation_CER")
    progress = dict(next_batch=0, global_step=0, loss_sum=0.0, loss_batches=0)
    start_epoch, best_val_cer, stale, history = 0, float("inf"), 0, []
    if checkpoint:
        for key in ("train_limit", "val_limit", "width_budget", "max_batch_size", "random_seed", "amp", "image_height", "hidden_size", "num_lstm_layers"):
            if checkpoint["config"].get(key) != config[key]:
                raise ValueError(f"Resume configuration mismatch: {key}")
        if checkpoint["num_classes"] != len(vocabulary) or checkpoint["sampler_state"] != sampler_state:
            raise ValueError("Resume class count, dataset identity or sampler widths differ")
        if checkpoint["architecture"] != dict(name="CRNN", hidden_size=HIDDEN_SIZE, num_lstm_layers=NUM_LSTM_LAYERS):
            raise ValueError("Resume architecture differs")
        if checkpoint["history"] and not BEST_CHECKPOINT.exists():
            raise FileNotFoundError("Resume after validation requires best.pt alongside last.pt")
        if BEST_CHECKPOINT.exists():
            previous_best = torch.load(BEST_CHECKPOINT, map_location="cpu", weights_only=True)
            if (previous_best['char_to_idx'] != checkpoint['char_to_idx'] or
                    previous_best['best_val_cer'] != checkpoint['best_val_cer']):
                raise ValueError("best.pt does not match resumed last.pt; restore a matching checkpoint pair")
            del previous_best
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"]
        progress = checkpoint["progress"]
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        best_val_cer = checkpoint["best_val_cer"]
        stale = checkpoint["epochs_without_improvement"]
        history = checkpoint["history"]
        random.setstate(checkpoint["random_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        if device.type == "cuda" and checkpoint["cuda_rng_states"]:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_states"])
        del checkpoint
        print(f"Resume: epoch={start_epoch} next_batch={progress['next_batch']} "
              f"global_step={progress['global_step']}", flush=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    # Probe both largest padded pixel count and longest sequence. Restore BN/RNG;
    # backward allocates gradients, and temporary optimizer step allocates Adam state.
    sampler.set_epoch(start_epoch, progress["next_batch"])
    def cpu_snapshot(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {key: cpu_snapshot(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cpu_snapshot(item) for item in value]
        return value
    saved_model = cpu_snapshot(model.state_dict())
    saved_optimizer = cpu_snapshot(optimizer.state_dict())
    cpu_rng, py_rng = torch.get_rng_state(), random.getstate()
    cuda_rng = torch.cuda.get_rng_state_all() if device.type == "cuda" else []
    candidates = {tuple(max(sampler.batches, key=lambda b: max(widths[i] for i in b) * len(b))),
                  tuple(max(sampler.batches, key=lambda b: max(widths[i] for i in b)))}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    try:
        probe_scaler = torch.amp.GradScaler("cuda", enabled=scaler.is_enabled())
        for indices in candidates:
            print(f"Memory probe: samples={len(indices)} max_width={max(widths[i] for i in indices)} "
                  f"padded_width={len(indices) * max(widths[i] for i in indices)}", flush=True)
            probe = [talim_collate_fn([train_dataset[i] for i in indices])]
            train_one_epoch(model, probe, encoder, criterion, optimizer, device, probe_scaler)
        print("Forward/backward/Adam memory preflight passed", flush=True)
    except torch.cuda.OutOfMemoryError as error:
        raise RuntimeError("Startup GPU OOM: lower --width-budget and/or --max-batch-size; no samples were dropped") from error
    finally:
        if device.type == "cuda":
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
            print(f"GPU memory GiB: peak_allocated={torch.cuda.max_memory_allocated(device) / 2**30:.2f} "
                  f"peak_reserved={torch.cuda.max_memory_reserved(device) / 2**30:.2f} "
                  f"free={free_bytes / 2**30:.2f} total={total_bytes / 2**30:.2f}", flush=True)
        model.load_state_dict(saved_model)
        optimizer.load_state_dict(saved_optimizer)
        model.zero_grad(set_to_none=True)
        del saved_model, saved_optimizer, probe_scaler
        random.setstate(py_rng)
        torch.set_rng_state(cpu_rng)
        if cuda_rng:
            torch.cuda.set_rng_state_all(cuda_rng)
            torch.cuda.empty_cache()
    if args.preflight_only:
        return
    invocation_start = progress["global_step"]
    for epoch in range(start_epoch, args.epochs):
        if stale >= EARLY_STOP_PATIENCE:
            break
        print(f"Epoch {epoch + 1}/{args.epochs}", flush=True)
        lr = optimizer.param_groups[0]["lr"]
        sampler.set_epoch(epoch, progress["next_batch"])
        print(f"Training continuation: epoch={epoch} next_batch={progress['next_batch']} "
              f"global_step={progress['global_step']} remaining_batches={len(sampler)}", flush=True)
        def persist(current):
            config["epochs_without_improvement"] = stale
            save_checkpoint(LAST_CHECKPOINT, model, optimizer, scheduler, epoch,
                            best_val_cer, vocabulary, train_characters, history, config,
                            scaler, current, sampler_state)
        def after_batch(current):
            stopping = args.stop_after_batches and current["global_step"] - invocation_start >= args.stop_after_batches
            if current["global_step"] % args.checkpoint_every_batches == 0 or stopping:
                persist(current)
            if stopping:
                raise StopIteration
        try:
            loss = train_one_epoch(model, train_loader, encoder, criterion, optimizer, device,
                                   scaler, progress, after_batch)
        except StopIteration:
            print("Saved mid-epoch last.pt; exiting before validation/test", flush=True)
            return
        metrics = evaluate(model, val_loader, encoder, criterion, vocabulary, device, "validation")
        print_evaluation("Validation", metrics)
        scheduler.step(metrics["cer"])
        history.append(dict(epoch=epoch + 1, learning_rate=lr, train_loss=loss, validation=metrics))
        improved = metrics["cer"] < best_val_cer
        if improved:
            best_val_cer, stale = metrics["cer"], 0
        else:
            stale += 1
        config["epochs_without_improvement"] = stale
        progress.update(next_batch=0, loss_sum=0.0, loss_batches=0)
        values = dict(scaler=scaler, progress=progress, sampler_state=sampler_state, model=model, optimizer=optimizer, scheduler=scheduler, epoch=epoch + 1,
                      best_val_cer=best_val_cer, vocabulary=vocabulary,
                      train_characters=train_characters, history=history, config=config)
        if improved:
            save_checkpoint(BEST_CHECKPOINT, **values)
        save_checkpoint(LAST_CHECKPOINT, **values)
        save_history(history)
    if args.skip_test:
        print("Preflight complete; TEST was not loaded or evaluated.", flush=True)
        return
    best = torch.load(BEST_CHECKPOINT, map_location="cpu", weights_only=True)
    model.load_state_dict(best["model_state_dict"])
    test_dataset = TalimTierDataset(TEST_PATH)
    test_widths, _ = resized_widths(test_dataset, OUTPUT_ROOT / "test_widths.json")
    test_loader = budget_loader(test_dataset, WidthBudgetBatchSampler(test_widths, args.width_budget, max_batch_size, args.seed))
    metrics = evaluate(model, test_loader, encoder, criterion, vocabulary, device, "test")
    print_evaluation("Test", metrics)
    TEST_REPORT.write_text(json.dumps(dict(best_epoch=best["epoch"],
                          best_validation_cer=best["best_val_cer"], test=metrics), indent=2), encoding="utf-8")


if __name__ == "__main__":
    import sys
    if any(flag.split('=')[0] in ('--train-manifest', '--val-manifest', '--total-steps', '--benchmark-steps') for flag in sys.argv[1:]):
        from training.manifest_training import main as manifest_main
        manifest_main()
    else:
        main()
