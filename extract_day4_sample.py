"""
NAQSHKASH
Rida - Day 4 helper

Creates a small training subset directly from talim_100000.zip
without extracting the entire 100,000-sample dataset.

It copies:
- first N training images
- matching TXT labels
- matching JSON files
- filtered metadata.csv

into:

Day4_sample/
└── train/
    ├── images/
    ├── labels/
    ├── json/
    └── metadata.csv
"""

from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

ZIP_PATH = Path(
    r"C:\FYP Dataset Generation\Final dataset\talim_100000.zip"
)

OUTPUT_ROOT = Path(
    r"C:\FYP Dataset Generation\Day4_sample\train"
)

NUM_SAMPLES = 100


# ============================================================
# HELPERS
# ============================================================

def find_member(
    names: list[str],
    suffix: str,
) -> str:
    """
    Find a ZIP member whose path ends with the supplied suffix.
    """
    suffix = suffix.replace("\\", "/")

    for name in names:
        if name.replace("\\", "/").endswith(suffix):
            return name

    raise FileNotFoundError(
        f"Could not find ZIP member ending with: {suffix}"
    )


def copy_member(
    zf: zipfile.ZipFile,
    member_name: str,
    destination: Path,
) -> None:
    """
    Copy one file from ZIP to disk.
    """
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with zf.open(member_name) as src:
        with destination.open("wb") as dst:
            dst.write(
                src.read()
            )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    if not ZIP_PATH.exists():
        raise FileNotFoundError(
            f"ZIP not found:\n{ZIP_PATH}"
        )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    images_dir = (
        OUTPUT_ROOT
        / "images"
    )

    labels_dir = (
        OUTPUT_ROOT
        / "labels"
    )

    json_dir = (
        OUTPUT_ROOT
        / "json"
    )

    images_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    labels_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(
        "Opening ZIP:"
    )

    print(
        ZIP_PATH
    )

    with zipfile.ZipFile(
        ZIP_PATH,
        "r",
    ) as zf:

        names = zf.namelist()

        # ----------------------------------------------------
        # Locate train metadata.csv
        # ----------------------------------------------------

        metadata_member = None

        for name in names:

            normalized = (
                name.replace(
                    "\\",
                    "/",
                )
            )

            if normalized.endswith(
                "/train/metadata.csv"
            ):

                metadata_member = name
                break

        if metadata_member is None:
            raise FileNotFoundError(
                "Could not find train/metadata.csv "
                "inside the ZIP."
            )

        print()
        print(
            "Found metadata:"
        )

        print(
            metadata_member
        )

        # ----------------------------------------------------
        # Read metadata directly from ZIP
        # ----------------------------------------------------

        raw = zf.read(
            metadata_member
        )

        text = raw.decode(
            "utf-8-sig"
        )

        reader = csv.DictReader(
            io.StringIO(
                text
            )
        )

        rows = list(
            reader
        )

        if not rows:
            raise ValueError(
                "metadata.csv contains no rows."
            )

        selected_rows = rows[
            :NUM_SAMPLES
        ]

        print()
        print(
            f"Selecting first "
            f"{len(selected_rows)} "
            f"training samples."
        )

        # ----------------------------------------------------
        # Detect useful metadata columns
        # ----------------------------------------------------

        fieldnames = (
            reader.fieldnames
        )

        if fieldnames is None:
            raise ValueError(
                "metadata.csv has no header."
            )

        # ----------------------------------------------------
        # Extract matching sample files
        # ----------------------------------------------------

        copied = 0

        for index, row in enumerate(
            selected_rows,
            start=1,
        ):

            sample_id = (
                row.get(
                    "sample_id"
                )
                or ""
            ).strip()

            if not sample_id:
                raise ValueError(
                    f"Metadata row {index} "
                    f"has no sample_id."
                )

            # -----------------------------------------------
            # IMAGE
            # -----------------------------------------------

            image_member = None

            for extension in (
                ".png",
                ".jpg",
                ".jpeg",
                ".bmp",
                ".tif",
                ".tiff",
            ):

                try:
                    image_member = (
                        find_member(
                            names,
                            f"/train/images/"
                            f"{sample_id}"
                            f"{extension}",
                        )
                    )

                    break

                except FileNotFoundError:
                    pass

            if image_member is None:
                raise FileNotFoundError(
                    f"Image not found for "
                    f"{sample_id}"
                )

            # -----------------------------------------------
            # LABEL
            # -----------------------------------------------

            label_member = find_member(
                names,
                f"/train/labels/"
                f"{sample_id}.txt",
            )

            # -----------------------------------------------
            # JSON
            # -----------------------------------------------

            json_member = find_member(
                names,
                f"/train/json/"
                f"{sample_id}.json",
            )

            # -----------------------------------------------
            # COPY
            # -----------------------------------------------

            image_name = Path(
                image_member
            ).name

            copy_member(
                zf,
                image_member,
                images_dir
                / image_name,
            )

            copy_member(
                zf,
                label_member,
                labels_dir
                / f"{sample_id}.txt",
            )

            copy_member(
                zf,
                json_member,
                json_dir
                / f"{sample_id}.json",
            )

            copied += 1

            print(
                f"{copied:03d}/"
                f"{len(selected_rows):03d} "
                f"{sample_id}"
            )

        # ----------------------------------------------------
        # Write filtered metadata.csv
        # ----------------------------------------------------

        metadata_output = (
            OUTPUT_ROOT
            / "metadata.csv"
        )

        with metadata_output.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
            )

            writer.writeheader()

            writer.writerows(
                selected_rows
            )

    print()
    print(
        "=" * 70
    )

    print(
        "Day 4 sample dataset created successfully."
    )

    print(
        f"Samples: {copied}"
    )

    print(
        f"Output: {OUTPUT_ROOT}"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()