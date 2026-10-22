"""
NAQSHKASH
Baseline CRNN + CTC model.

Input:
    [B, 1, 64, W]

CNN:
    extracts visual features
    reduces height 64 -> 1
    reduces width approximately x4

BiLSTM:
    reads features left-to-right and right-to-left

Classifier:
    predicts Talim character classes + CTC blank
"""

import torch
import torch.nn as nn


class CRNN(nn.Module):

    def __init__(
        self,
        num_classes,
        hidden_size=256,
        num_lstm_layers=2,
    ):

        super().__init__()

        # ====================================================
        # CNN FEATURE EXTRACTOR
        # ====================================================

        self.cnn = nn.Sequential(

            # ------------------------------------------------
            # 64 x W
            # ------------------------------------------------

            nn.Conv2d(
                1,
                64,
                kernel_size=3,
                padding=1,
            ),

            nn.BatchNorm2d(
                64
            ),

            nn.ReLU(
                inplace=True
            ),

            # 64xW -> 32x(W/2)
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),

            # ------------------------------------------------
            # 32 x W/2
            # ------------------------------------------------

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1,
            ),

            nn.BatchNorm2d(
                128
            ),

            nn.ReLU(
                inplace=True
            ),

            # 32xW/2 -> 16xW/4
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),

            # ------------------------------------------------
            # 16 x W/4
            # ------------------------------------------------

            nn.Conv2d(
                128,
                256,
                kernel_size=3,
                padding=1,
            ),

            nn.BatchNorm2d(
                256
            ),

            nn.ReLU(
                inplace=True
            ),

            # Only reduce HEIGHT now.
            nn.MaxPool2d(
                kernel_size=(2, 1),
                stride=(2, 1),
            ),

            # ------------------------------------------------
            # 8 x W/4
            # ------------------------------------------------

            nn.Conv2d(
                256,
                256,
                kernel_size=3,
                padding=1,
            ),

            nn.BatchNorm2d(
                256
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.MaxPool2d(
                kernel_size=(2, 1),
                stride=(2, 1),
            ),

            # ------------------------------------------------
            # 4 x W/4
            # ------------------------------------------------

            nn.Conv2d(
                256,
                512,
                kernel_size=3,
                padding=1,
            ),

            nn.BatchNorm2d(
                512
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.MaxPool2d(
                kernel_size=(2, 1),
                stride=(2, 1),
            ),

            # ------------------------------------------------
            # 2 x W/4
            # ------------------------------------------------

            nn.Conv2d(
                512,
                512,
                kernel_size=3,
                padding=1,
            ),

            nn.BatchNorm2d(
                512
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.MaxPool2d(
                kernel_size=(2, 1),
                stride=(2, 1),
            ),

            # ------------------------------------------------
            # Final:
            # height = 1
            # width ~= W/4
            # ------------------------------------------------
        )

        # ====================================================
        # BIDIRECTIONAL LSTM
        # ====================================================

        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=(
                0.1
                if num_lstm_layers > 1
                else 0.0
            ),
        )

        # ====================================================
        # CHARACTER CLASSIFIER
        # ====================================================

        self.classifier = nn.Linear(
            hidden_size * 2,
            num_classes,
        )


    def forward(
        self,
        images,
    ):
        """
        Parameters
        ----------
        images:
            [B, 1, 64, W]

        Returns
        -------
        logits:
            [T, B, C]

        CTC expects time dimension first.
        """

        features = self.cnn(
            images
        )

        # Expected:
        # [B, 512, 1, T]

        if features.shape[2] != 1:

            raise RuntimeError(
                "CNN output height must be 1. "
                "Got shape: {}".format(
                    tuple(
                        features.shape
                    )
                )
            )

        # Remove height.
        features = features.squeeze(
            2
        )

        # [B, 512, T]
        #
        # -> [B, T, 512]

        features = features.permute(
            0,
            2,
            1,
        )

        sequence, _ = self.rnn(
            features
        )

        # [B, T, C]

        logits = self.classifier(
            sequence
        )

        # CTCLoss expects:
        #
        # [T, B, C]

        logits = logits.permute(
            1,
            0,
            2,
        )

        return logits


    @staticmethod
    def output_lengths(
        image_widths,
    ):
        """
        CNN downsamples width by x4:

            pool x2
            pool x2

        Remaining pools affect height only.
        """

        lengths = (
            image_widths
            // 2
            // 2
        )

        return torch.clamp(
            lengths,
            min=1,
        )