"""
NAQSHKASH

Tiny CRNN + CTC overfit experiment.

Purpose
-------
Before large-scale training, deliberately train on a very small
number of tier samples.

If the model can memorize these samples, the training pipeline is
probably wired correctly.

This is NOT final training.
"""

import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import (
    DataLoader,
    Subset,
)

from data.tier_dataset import (
    TalimTierDataset,
)

from data.collate import (
    talim_collate_fn,
)

from data.vocabulary import (
    build_vocabulary_from_labels,
)

from data.encoder import (
    CTCLabelEncoder,
)

from data.ctc_decode import (
    greedy_ctc_decode,
)

from models.crnn import (
    CRNN,
)


# ============================================================
# CONFIG
# ============================================================

DATASET_PATH = Path(
    "datasets/Day4_tier_dataset/train"
)

NUM_SAMPLES = 32

BATCH_SIZE = 8

MAX_EPOCHS = 200

LEARNING_RATE = 0.001

RANDOM_SEED = 42

EVALUATE_EVERY = 5

# Once this accuracy is reached, tiny overfit test is successful.
PASS_ACCURACY = 0.95

CHECKPOINT_DIR = Path(
    "checkpoints"
)

BEST_CHECKPOINT_PATH = (
    CHECKPOINT_DIR
    / "best_overfit_crnn.pt"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(
    RANDOM_SEED
)

torch.manual_seed(
    RANDOM_SEED
)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(
        RANDOM_SEED
    )


# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():

    DEVICE = torch.device(
        "cuda"
    )

else:

    DEVICE = torch.device(
        "cpu"
    )


# ============================================================
# DECODE MODEL OUTPUT
# ============================================================

def decode_predictions(
    logits,
    input_lengths,
    vocabulary,
):

    predictions = torch.argmax(
        logits,
        dim=2,
    )

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

        decoded.append(
            text
        )

    return decoded


# ============================================================
# FULL SUBSET EVALUATION
# ============================================================

def evaluate_exact_match(
    model,
    loader,
    vocabulary,
):

    model.eval()

    total_samples = 0
    exact_matches = 0

    results = []

    with torch.no_grad():

        for batch in loader:

            images = (
                batch[
                    "images"
                ]
                .to(
                    DEVICE
                )
            )

            image_widths = (
                batch[
                    "image_widths"
                ]
                .to(
                    DEVICE
                )
            )

            logits = model(
                images
            )

            input_lengths = (
                model.output_lengths(
                    image_widths
                )
            )

            predictions = decode_predictions(
                logits,
                input_lengths,
                vocabulary,
            )

            truths = (
                batch[
                    "label_texts"
                ]
            )

            sample_ids = (
                batch[
                    "sample_ids"
                ]
            )

            for (
                sample_id,
                truth,
                predicted,
            ) in zip(
                sample_ids,
                truths,
                predictions,
            ):

                correct = (
                    predicted
                    ==
                    truth
                )

                if correct:

                    exact_matches += 1

                total_samples += 1

                results.append(
                    {
                        "sample_id":
                            sample_id,

                        "truth":
                            truth,

                        "prediction":
                            predicted,

                        "correct":
                            correct,
                    }
                )

    accuracy = (
        exact_matches
        /
        total_samples
        if total_samples > 0
        else 0.0
    )

    return {
        "exact_matches":
            exact_matches,

        "total_samples":
            total_samples,

        "accuracy":
            accuracy,

        "results":
            results,
    }


# ============================================================
# PRINT PREVIEW
# ============================================================

def print_prediction_preview(
    evaluation,
    max_examples=6,
):

    print(
        "-" * 75
    )

    for result in evaluation[
        "results"
    ][:max_examples]:

        print(
            "ID:   ",
            result[
                "sample_id"
            ],
        )

        print(
            "TRUE: ",
            repr(
                result[
                    "truth"
                ]
            ),
        )

        print(
            "PRED: ",
            repr(
                result[
                    "prediction"
                ]
            ),
        )

        print(
            "MATCH:",
            (
                "YES"
                if result[
                    "correct"
                ]
                else "NO"
            ),
        )

        print()

    print(
        "-" * 75
    )


# ============================================================
# SAVE CHECKPOINT
# ============================================================

def save_checkpoint(
    model,
    optimizer,
    epoch,
    accuracy,
    exact_matches,
    vocabulary,
):

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "epoch":
            epoch,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "accuracy":
            accuracy,

        "exact_matches":
            exact_matches,

        "num_samples":
            NUM_SAMPLES,

        "vocabulary_size":
            len(
                vocabulary
            ),

        "blank_index":
            vocabulary.blank_index,
    }

    torch.save(
        checkpoint,
        BEST_CHECKPOINT_PATH,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()

    print(
        "=" * 75
    )

    print(
        "NAQSHKASH - TINY CRNN OVERFIT TEST"
    )

    print(
        "Device:",
        DEVICE,
    )

    print(
        "=" * 75
    )

    # ========================================================
    # DATASET
    # ========================================================

    full_dataset = TalimTierDataset(
        DATASET_PATH
    )

    if len(full_dataset) < NUM_SAMPLES:

        raise ValueError(
            "Dataset contains fewer than "
            f"{NUM_SAMPLES} samples."
        )

    subset_indices = list(
        range(
            NUM_SAMPLES
        )
    )

    subset = Subset(
        full_dataset,
        subset_indices,
    )

    train_loader = DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=talim_collate_fn,
    )

    eval_loader = DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=talim_collate_fn,
    )

    print(
        "Training samples:",
        len(
            subset
        ),
    )

    # ========================================================
    # VOCABULARY
    # ========================================================

    vocabulary = (
        build_vocabulary_from_labels(
            DATASET_PATH
            / "labels"
        )
    )

    encoder = CTCLabelEncoder(
        vocabulary
    )

    print(
        "Vocabulary size:",
        len(
            vocabulary
        ),
    )

    print(
        "CTC blank:",
        vocabulary.blank_index,
    )

    # ========================================================
    # MODEL
    # ========================================================

    model = CRNN(
        num_classes=len(
            vocabulary
        ),
        hidden_size=256,
        num_lstm_layers=2,
    )

    model = model.to(
        DEVICE
    )

    # ========================================================
    # LOSS
    # ========================================================

    ctc_loss = nn.CTCLoss(
        blank=vocabulary.blank_index,
        reduction="mean",
        zero_infinity=True,
    )

    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # ========================================================
    # BEST RESULT TRACKING
    # ========================================================

    best_accuracy = 0.0
    best_epoch = 0
    best_exact_matches = 0

    passed = False

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):

        model.train()

        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:

            images = (
                batch[
                    "images"
                ]
                .to(
                    DEVICE
                )
            )

            image_widths = (
                batch[
                    "image_widths"
                ]
                .to(
                    DEVICE
                )
            )

            label_texts = (
                batch[
                    "label_texts"
                ]
            )

            # ------------------------------------------------
            # Encode labels
            # ------------------------------------------------

            encoded = (
                encoder.encode_batch(
                    label_texts
                )
            )

            targets = (
                encoded[
                    "targets"
                ]
                .to(
                    DEVICE
                )
            )

            target_lengths = (
                encoded[
                    "target_lengths"
                ]
                .to(
                    DEVICE
                )
            )

            # ------------------------------------------------
            # Forward
            # ------------------------------------------------

            logits = model(
                images
            )

            log_probs = torch.log_softmax(
                logits,
                dim=2,
            )

            input_lengths = (
                model.output_lengths(
                    image_widths
                )
            )

            if torch.any(
                target_lengths
                >
                input_lengths
            ):

                raise RuntimeError(
                    "CTC target length exceeds "
                    "available sequence length."
                )

            # ------------------------------------------------
            # Loss
            # ------------------------------------------------

            loss = ctc_loss(
                log_probs,
                targets,
                input_lengths,
                target_lengths,
            )

            # ------------------------------------------------
            # Backprop
            # ------------------------------------------------

            optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )

            optimizer.step()

            total_loss += float(
                loss.item()
            )

            num_batches += 1

        average_loss = (
            total_loss
            /
            max(
                1,
                num_batches,
            )
        )

        # ====================================================
        # EVALUATION
        # ====================================================

        should_evaluate = (
            epoch == 1
            or epoch % EVALUATE_EVERY == 0
            or epoch == MAX_EPOCHS
        )

        if not should_evaluate:

            continue

        evaluation = evaluate_exact_match(
            model,
            eval_loader,
            vocabulary,
        )

        exact_matches = (
            evaluation[
                "exact_matches"
            ]
        )

        total_samples = (
            evaluation[
                "total_samples"
            ]
        )

        accuracy = (
            evaluation[
                "accuracy"
            ]
        )

        print()

        print(
            (
                "Epoch {:03d} | "
                "Loss {:.4f} | "
                "Exact {}/{} ({:.1f}%)"
            ).format(
                epoch,
                average_loss,
                exact_matches,
                total_samples,
                accuracy * 100.0,
            )
        )

        print_prediction_preview(
            evaluation
        )

        # ====================================================
        # SAVE BEST MODEL
        # ====================================================

        if accuracy > best_accuracy:

            best_accuracy = (
                accuracy
            )

            best_epoch = (
                epoch
            )

            best_exact_matches = (
                exact_matches
            )

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                accuracy=accuracy,
                exact_matches=exact_matches,
                vocabulary=vocabulary,
            )

            print(
                "Best checkpoint saved:"
            )

            print(
                BEST_CHECKPOINT_PATH
            )

        # ====================================================
        # EARLY STOP
        # ====================================================

        if accuracy >= PASS_ACCURACY:

            passed = True

            print()

            print(
                "STATUS: OVERFIT TARGET REACHED"
            )

            print(
                "Early stopping."
            )

            break

        else:

            print(
                "STATUS: still learning"
            )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()

    print(
        "=" * 75
    )

    print(
        "OVERFIT TEST FINISHED"
    )

    print(
        "=" * 75
    )

    print(
        "Best exact matches:",
        f"{best_exact_matches}/{NUM_SAMPLES}",
    )

    print(
        "Best accuracy:",
        "{:.1f}%".format(
            best_accuracy
            *
            100.0
        ),
    )

    print(
        "Best epoch:",
        best_epoch,
    )

    print(
        "Best checkpoint:",
        BEST_CHECKPOINT_PATH,
    )

    print()

    if passed:

        print(
            "RESULT: PASS"
        )

        print(
            "The CRNN + CTC pipeline successfully "
            "memorized the tiny training subset."
        )

    else:

        print(
            "RESULT: NOT PASSED"
        )

        print(
            "The model did not reach the required "
            "tiny-subset exact-match accuracy."
        )

    print(
        "=" * 75
    )


if __name__ == "__main__":

    main()