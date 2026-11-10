"""
NAQSHKASH
CTC feasibility analysis for tier-level OCR samples.

Checks whether each target can theoretically fit
inside the model's horizontal time sequence.
"""

from pathlib import Path

from data.tier_dataset import (
    TalimTierDataset,
)

from data.collate import (
    resize_preserve_aspect_ratio,
)


DATASET_PATH = Path(
    "datasets/Day4_tier_dataset/train"
)

TARGET_HEIGHT = 64


def minimum_ctc_steps(
    text,
):
    """
    CTC needs an extra blank time-step between
    adjacent identical target symbols.

    Example:
        "AAA"

    needs at least:
        A blank A blank A

    = 5 time steps.
    """

    if not text:
        return 0

    repeated_pairs = 0

    for i in range(
        1,
        len(text),
    ):

        if (
            text[i]
            ==
            text[i - 1]
        ):

            repeated_pairs += 1

    return (
        len(text)
        + repeated_pairs
    )


def main():

    dataset = TalimTierDataset(
        DATASET_PATH
    )

    factors = [
        1,
        2,
        4,
        8,
    ]

    failures = {
        factor: []
        for factor in factors
    }

    smallest_margin = {
        factor: None
        for factor in factors
    }

    print()
    print(
        "=" * 75
    )

    print(
        "CTC FEASIBILITY ANALYSIS"
    )

    print(
        f"Samples: {len(dataset)}"
    )

    print(
        f"Target height: "
        f"{TARGET_HEIGHT}"
    )

    print(
        "=" * 75
    )

    for i in range(
        len(dataset)
    ):

        sample = dataset[i]

        resized = (
            resize_preserve_aspect_ratio(
                sample["image"],
                target_height=TARGET_HEIGHT,
            )
        )

        width = (
            resized.shape[-1]
        )

        text = (
            sample[
                "label_text"
            ]
        )

        required_steps = (
            minimum_ctc_steps(
                text
            )
        )

        for factor in factors:

            # Conservative approximation:
            # CNN reduces width by this factor.
            available_steps = max(
                1,
                width // factor,
            )

            margin = (
                available_steps
                - required_steps
            )

            if (
                smallest_margin[
                    factor
                ]
                is None
                or margin
                <
                smallest_margin[
                    factor
                ]
            ):

                smallest_margin[
                    factor
                ] = margin

            if (
                available_steps
                <
                required_steps
            ):

                failures[
                    factor
                ].append(
                    {
                        "tier_id":
                            sample[
                                "tier_id"
                            ],

                        "tier":
                            sample[
                                "tier_name"
                            ],

                        "width":
                            width,

                        "target":
                            text,

                        "target_length":
                            len(text),

                        "required_steps":
                            required_steps,

                        "available_steps":
                            available_steps,
                    }
                )

    print()

    for factor in factors:

        count = len(
            failures[
                factor
            ]
        )

        percentage = (
            100.0
            * count
            / len(dataset)
        )

        print(
            f"Horizontal downsample x{factor}:"
        )

        print(
            f"  Failures: "
            f"{count}/{len(dataset)} "
            f"({percentage:.2f}%)"
        )

        print(
            f"  Worst margin: "
            f"{smallest_margin[factor]}"
        )

        if count:

            print(
                "  First few failures:"
            )

            for failure in (
                failures[
                    factor
                ][:5]
            ):

                print(
                    "   ",
                    failure,
                )

        print()

    print(
        "=" * 75
    )


if __name__ == "__main__":
    main()