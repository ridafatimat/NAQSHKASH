"""
data.vocabulary
===============
Token vocabulary and character mappings for Kashmiri Carpet Talim OCR.

Author: Ayesha Amer (Day 4 Scope)
Project: NAQSHKASH FYP

Design Contract:
----------------
- Canonical Alphabet:
  * 30 Trusted Symbols: 16 Upper ('A'-'N', 'P', 'Q') + 14 Lower ('R'-'Z', '[', '\\', ']', '^', '_', 'W')
  * 34 Authentic Count Glyphs: 'a'-'z', '{', '|', "'", '(', ')', '*', '+', ','
  * 2 Formatting Tokens: ' ' (space), '\n' (newline separator for 3-tier logical rows)
- CTC Blank Token:
  * Reserved at index 0 (<blank>).
  * 0 is NEVER assigned to any character in `char_to_index`.
  * `len(vocabulary)` returns total classes (characters + 1 for CTC blank = 67).
- Deterministic 1-based indexing for characters: 1, 2, ..., N.
"""

from pathlib import Path
from typing import List, Dict, Union, Optional, Set, Iterable, Any
import json
import glob
import os


# Canonical 30 Trusted Symbols (16 Upper, 14 Lower)
CANONICAL_UPPER_SYMBOLS: List[str] = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "P", "Q"
]

CANONICAL_LOWER_SYMBOLS: List[str] = [
    "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "[", "\\", "]", "^", "_"
]

CANONICAL_SYMBOLS: List[str] = sorted(CANONICAL_UPPER_SYMBOLS + CANONICAL_LOWER_SYMBOLS)

# Canonical 34 Authentic Count Glyphs (1-259 range in Talamat font)
CANONICAL_COUNT_GLYPHS: List[str] = [
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j",
    "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
    "u", "v", "w", "x", "y", "z", "{", "|",
    "'", "(", ")", "*", "+", ","
]

# Formatting tokens
FORMATTING_CHARS: List[str] = [" ", "\n"]

# Complete canonical character set
CANONICAL_CHARACTERS: List[str] = sorted(
    list(set(CANONICAL_SYMBOLS + CANONICAL_COUNT_GLYPHS + FORMATTING_CHARS))
)


def normalize_label_text(text: str) -> str:
    """
    Normalize line breaks in label text to standard '\\n'.
    
    Parameters
    ----------
    text : str
        Raw label text string.
        
    Returns
    -------
    str
        Normalized label text with all Windows (\\r\\n) and Mac (\\r) line breaks converted to \\n.
    """
    if not isinstance(text, str):
        raise TypeError(f"Expected str, got {type(text)}")
    return text.replace("\r\n", "\n").replace("\r", "\n")


class TalimVocabulary:
    """
    Character vocabulary and bidirectional token-to-index mapping for Talim OCR.
    
    Attributes
    ----------
    blank_index : int
        Index reserved for the CTC blank token (always 0).
    blank_token : str
        String representation of the CTC blank token (default '<blank>').
    char_to_index : Dict[str, int]
        Mapping from character string to 1-based integer token ID.
    index_to_char : Dict[int, str]
        Mapping from integer token ID to character string (index 0 maps to blank_token).
    characters : List[str]
        List of unique characters in the vocabulary.
    """

    def __init__(
        self,
        characters: Optional[Iterable[str]] = None,
        blank_index: int = 0,
        blank_token: str = "<blank>",
    ):
        self.blank_index: int = blank_index
        self.blank_token: str = blank_token

        if characters is None:
            chars = list(CANONICAL_CHARACTERS)
        else:
            # Deterministic unique list
            seen = set()
            chars = []
            for c in characters:
                if c not in seen and c != blank_token:
                    seen.add(c)
                    chars.append(c)

        self.characters: List[str] = chars

        # 1-based indexing for actual characters
        self.char_to_index: Dict[str, int] = {}
        self.index_to_char: Dict[int, str] = {self.blank_index: self.blank_token}

        current_idx = 1
        for char in self.characters:
            if current_idx == self.blank_index:
                current_idx += 1
            self.char_to_index[char] = current_idx
            self.index_to_char[current_idx] = char
            current_idx += 1

    def __len__(self) -> int:
        """Return total vocabulary size including the CTC blank token."""
        return len(self.char_to_index) + 1

    def encode(self, text: str) -> List[int]:
        """
        Encode a text string into a list of integer token IDs.
        
        Parameters
        ----------
        text : str
            Input label text string.
            
        Returns
        -------
        List[int]
            List of integer token IDs.
            
        Raises
        ------
        KeyError
            If any character in `text` is not in the vocabulary.
        """
        normalized = normalize_label_text(text)
        token_ids: List[int] = []
        for i, char in enumerate(normalized):
            if char not in self.char_to_index:
                raise KeyError(
                    f"Unknown character {repr(char)} (Unicode U+{ord(char):04X}) "
                    f"at position {i} in text: {repr(text[:50])}"
                )
            token_ids.append(self.char_to_index[char])
        return token_ids

    def decode(
        self,
        token_ids: Iterable[int],
        remove_blank: bool = True,
    ) -> str:
        """
        Decode a sequence of integer token IDs back into a text string.
        
        Parameters
        ----------
        token_ids : Iterable[int]
            Sequence of integer token IDs.
        remove_blank : bool, default=True
            Whether to skip the CTC blank index (0) during decoding.
            
        Returns
        -------
        str
            Decoded text string.
        """
        chars: List[str] = []
        for tid in token_ids:
            idx = int(tid)
            if remove_blank and idx == self.blank_index:
                continue
            if idx in self.index_to_char:
                c = self.index_to_char[idx]
                if c != self.blank_token:
                    chars.append(c)
            else:
                raise KeyError(f"Unknown token ID {idx} for vocabulary of size {len(self)}")
        return "".join(chars)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize vocabulary configuration to a dictionary."""
        return {
            "blank_index": self.blank_index,
            "blank_token": self.blank_token,
            "characters": self.characters,
            "char_to_index": self.char_to_index,
            "size": len(self),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TalimVocabulary":
        """Reconstruct TalimVocabulary from serialized dictionary."""
        return cls(
            characters=data["characters"],
            blank_index=data.get("blank_index", 0),
            blank_token=data.get("blank_token", "<blank>"),
        )

    def save(self, file_path: Union[str, Path]) -> None:
        """Save vocabulary mapping to a JSON file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> "TalimVocabulary":
        """Load vocabulary from a JSON file."""
        path = Path(file_path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


def build_vocabulary_from_labels(
    labels_path: Optional[Union[str, Path]] = None,
) -> TalimVocabulary:
    """
    Build a TalimVocabulary instance from label text files or canonical fallback.
    
    If `labels_path` is provided and contains `.txt` files, all unique characters across
    the files are extracted (guaranteeing ' ' and '\\n' are preserved). If `labels_path`
    is None or contains no label files, the canonical 66-character vocabulary is used.
    
    Parameters
    ----------
    labels_path : Optional[Union[str, Path]], default=None
        Path to directory containing ground-truth `.txt` label files.
        
    Returns
    -------
    TalimVocabulary
        Initialized vocabulary instance with CTC blank at index 0.
    """
    if labels_path is not None:
        path = Path(labels_path)
        if path.is_file() and path.suffix == ".txt":
            label_files = [str(path)]
        elif path.is_dir():
            label_files = glob.glob(str(path / "**" / "*.txt"), recursive=True)
            if not label_files:
                label_files = glob.glob(str(path / "*.txt"))
        else:
            label_files = []

        if label_files:
            unique_chars: Set[str] = set()
            for lf in label_files:
                try:
                    with open(lf, "r", encoding="utf-8", errors="ignore") as f:
                        text = normalize_label_text(f.read())
                        unique_chars.update(text)
                except Exception:
                    continue

            if unique_chars:
                # Always ensure newline and space are present for multi-tier Talim notation
                unique_chars.add(" ")
                unique_chars.add("\n")
                # Deterministic sort
                sorted_chars = sorted(list(unique_chars))
                return TalimVocabulary(characters=sorted_chars)

    # Fallback to canonical full vocabulary
    return TalimVocabulary(characters=CANONICAL_CHARACTERS)
