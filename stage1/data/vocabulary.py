"""
NAQSHKASH
Rida - Label Vocabulary for CTC OCR

Builds a character-level vocabulary from Talim TXT labels.

Important:
- CTC blank token is always index 0.
- Real Talim label characters are assigned indices 1..N.
- Spaces and newline characters are preserved.
- Windows CRLF newlines are normalized to '\n'.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List


CTC_BLANK_TOKEN = "<BLANK>"
CTC_BLANK_INDEX = 0


def normalize_label_text(
    text: str,
) -> str:
    """
    Normalize line endings while preserving label structure.

    Converts:
        \\r\\n -> \\n
        \\r   -> \\n

    Does NOT strip spaces or internal newlines.
    """

    return (
        text
        .replace("\r\n", "\n")
        .replace("\r", "\n")
    )


class TalimVocabulary:
    """
    Character-level vocabulary for Talim OCR.

    Index 0 is reserved for the CTC blank token.
    """

    def __init__(
        self,
        characters: Iterable[str],
    ):

        unique_characters = sorted(
            set(characters)
        )

        if CTC_BLANK_TOKEN in unique_characters:
            raise ValueError(
                "CTC blank token must not appear "
                "inside real label characters."
            )

        self.blank_token = (
            CTC_BLANK_TOKEN
        )

        self.blank_index = (
            CTC_BLANK_INDEX
        )

        # Real label characters start at 1.
        self.char_to_index: Dict[str, int] = {
            char: index
            for index, char in enumerate(
                unique_characters,
                start=1,
            )
        }

        self.index_to_char: Dict[int, str] = {
            index: char
            for char, index in (
                self.char_to_index.items()
            )
        }

        # Index 0 is special.
        self.index_to_char[
            self.blank_index
        ] = self.blank_token


    def __len__(
        self,
    ) -> int:
        """
        Total number of classes including CTC blank.
        """

        return (
            len(
                self.char_to_index
            )
            + 1
        )


    @property
    def characters(
        self,
    ) -> List[str]:

        return list(
            self.char_to_index.keys()
        )


    def encode(
        self,
        text: str,
    ) -> List[int]:
        """
        Convert Talim label text to integer IDs.
        """

        text = normalize_label_text(
            text
        )

        encoded = []

        for char in text:

            if char not in self.char_to_index:

                display = repr(
                    char
                )

                raise KeyError(
                    f"Unknown character {display}. "
                    "Character is not present "
                    "in the vocabulary."
                )

            encoded.append(
                self.char_to_index[
                    char
                ]
            )

        return encoded


    def decode(
        self,
        indices: Iterable[int],
    ) -> str:
        """
        Convert non-CTC integer IDs back to text.

        Blank tokens are ignored.
        """

        characters = []

        for index in indices:

            index = int(
                index
            )

            if index == self.blank_index:
                continue

            if index not in self.index_to_char:

                raise KeyError(
                    f"Unknown vocabulary index: {index}"
                )

            characters.append(
                self.index_to_char[
                    index
                ]
            )

        return "".join(
            characters
        )


def build_vocabulary_from_labels(
    labels_dir: str | Path,
) -> TalimVocabulary:
    """
    Scan all .txt label files and derive the vocabulary.
    """

    labels_dir = Path(
        labels_dir
    )

    if not labels_dir.exists():

        raise FileNotFoundError(
            f"Labels directory not found:\n"
            f"{labels_dir}"
        )

    label_files = sorted(
        labels_dir.glob(
            "*.txt"
        )
    )

    if not label_files:

        raise ValueError(
            f"No TXT label files found in:\n"
            f"{labels_dir}"
        )

    characters = set()

    for label_path in label_files:

        text = label_path.read_text(
            encoding="utf-8",
        )

        text = normalize_label_text(
            text
        )

        # Remove only trailing file newline(s).
        #
        # We do NOT call .strip()
        # because spaces can be meaningful.
        text = text.rstrip(
            "\n"
        )

        characters.update(
            text
        )

    return TalimVocabulary(
        characters
    )