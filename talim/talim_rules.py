"""
NAQSHKASH
talim_rules.py

Single source of truth for Talim rules used by NAQSHKASH.

REFERENCE SOURCES
-----------------
The values in this module are derived from the original Talamat codebase
located at:

    vendored Talamat references in talim/reference/

Reference files:

    config.py
        -> UP_SYMBOLS
        -> DOWN_SYMBOLS
        -> SYMBOLS

    Utils/talamNumConverter.py
        -> num_mapping
        -> getTalamCountString()

    Logic/ImageToTalam.py
        -> three-line Talim row structure
        -> UP / DOWN placement
        -> horizontal rendering/alignment logic

    Docs/Font.txt
        -> Talamat glyph / count mapping confirmation

IMPORTANT
---------
Do not invent Talim rules in this file.

If domain-expert confirmation changes a rule later, update it here so the
dataset generation, validation and decoding layers can use the same source
of truth.
"""

from __future__ import annotations

import hashlib
import json
import runpy
from pathlib import Path
from typing import Dict, FrozenSet, List, Set, Tuple


# ============================================================================
# REFERENCE CODE LOCATION
# ============================================================================

REFERENCE_ROOT = Path(__file__).resolve().parent / "reference"


CONFIG_PATH = (
    REFERENCE_ROOT
    / "config.py"
)

COUNT_CONVERTER_PATH = (
    REFERENCE_ROOT
    / "Utils"
    / "talamNumConverter.py"
)


def _validate_reference_files() -> None:
    """
    Ensure the original Talamat reference files exist.

    We intentionally keep the Talim rules connected to the original
    implementation during development so that no mappings are manually
    re-created or accidentally changed.
    """

    required_files = [
        CONFIG_PATH,
        COUNT_CONVERTER_PATH,
    ]

    missing = [
        path
        for path in required_files
        if not path.exists()
    ]

    if missing:

        missing_text = "\n".join(
            str(path)
            for path in missing
        )

        raise FileNotFoundError(
            "Required Talamat reference files were not found:\n"
            f"{missing_text}"
        )


_validate_reference_files()


# ============================================================================
# LOAD ORIGINAL TALAMAT RULES
# ============================================================================

# runpy is used instead of importing the original modules as a package because
# parts of the original Talamat project may initialize GUI/application code.

_config = runpy.run_path(
    str(
        CONFIG_PATH
    )
)

_converter = runpy.run_path(
    str(
        COUNT_CONVERTER_PATH
    )
)


# ============================================================================
# SYMBOL RULES
# ============================================================================

UP_SYMBOLS: List[str] = list(
    _config[
        "UP_SYMBOLS"
    ]
)

DOWN_SYMBOLS: List[str] = list(
    _config[
        "DOWN_SYMBOLS"
    ]
)


# Symbols that appear in both source lists.
#
# Their direction cannot safely be inferred from membership alone.
AMBIGUOUS_SYMBOLS: FrozenSet[str] = frozenset(
    set(
        UP_SYMBOLS
    )
    &
    set(
        DOWN_SYMBOLS
    )
)


EXCLUSIVE_UP_SYMBOLS: List[str] = [
    symbol
    for symbol in UP_SYMBOLS
    if symbol not in AMBIGUOUS_SYMBOLS
]


EXCLUSIVE_DOWN_SYMBOLS: List[str] = [
    symbol
    for symbol in DOWN_SYMBOLS
    if symbol not in AMBIGUOUS_SYMBOLS
]


# Symbols whose direction can be determined unambiguously.
TRUSTED_SYMBOLS: FrozenSet[str] = frozenset(
    EXCLUSIVE_UP_SYMBOLS
    +
    EXCLUSIVE_DOWN_SYMBOLS
)


# All source-defined symbols, including ambiguous ones.
ALL_SYMBOLS: List[str] = sorted(
    set(
        UP_SYMBOLS
        +
        DOWN_SYMBOLS
    )
)


VALID_DIRECTIONS: Tuple[str, str] = (
    "UP",
    "DOWN",
)


# ============================================================================
# COUNT ENCODING
# ============================================================================

# Exact count mapping from:
#
# Utils/talamNumConverter.py -> num_mapping

COUNT_ENCODING: Dict[int, str] = dict(
    _converter[
        "num_mapping"
    ]
)


_original_encode_count = (
    _converter[
        "getTalamCountString"
    ]
)


MIN_AUTHENTIC_COUNT: int = 1


# The original converter includes a base glyph for 250 and then combines
# it with unit glyphs to represent values through 259.
MAX_AUTHENTIC_COUNT: int = (
    max(
        COUNT_ENCODING
    )
    +
    9
)


# Every count glyph explicitly present in the authentic mapping.
VALID_COUNT_GLYPHS: FrozenSet[str] = frozenset(
    glyph
    for count, glyph in COUNT_ENCODING.items()
    if count > 0
)


# ============================================================================
# RENDERING / FONT INFORMATION
# ============================================================================

FONT_NAME: str = "IMTCRPET"

FONT_FILE: str = (
    "Assets/Talamat.ttf"
)

DOCX_FONT_SIZE_PT: int = 9

PNG_DEFAULT_FONT_SIZE: int = 48


# ============================================================================
# COUNT API
# ============================================================================

def validate_count(
    count: int,
) -> None:
    """
    Validate an authentic Talamat run count.

    Authentic supported range:
        1 .. 259
    """

    if (
        type(count) is not int
        or not (
            MIN_AUTHENTIC_COUNT
            <= count
            <= MAX_AUTHENTIC_COUNT
        )
    ):

        raise ValueError(
            f"Count {count} is outside the authentic Talamat range "
            f"{MIN_AUTHENTIC_COUNT}-{MAX_AUTHENTIC_COUNT}."
        )


def encode_count(
    count: int,
) -> str:
    """
    Encode a Talim run length using the original Talamat converter.

    Production-safe function.

    Only authentic counts in the range 1-259 are accepted.

    Examples depend on the original mapping defined in
    talamNumConverter.py.
    """

    validate_count(
        count
    )

    return _original_encode_count(
        count
    )


def encode_count_extended(
    count: int,
) -> str:
    """
    Compatibility encoder for historical FYP tests only.

    WARNING
    -------
    Do NOT use this function for production Talim dataset generation.

    Older tests expected '?' to represent a 300 block, but '?' is not part
    of the confirmed authentic Talamat count-font mapping.

    Examples expected by the legacy test suite:

        299 -> "'ri"
        300 -> "?"
        301 -> "?a"
        600 -> "??"
    """

    if (
        type(count) is not int
        or count <= 0
    ):

        raise ValueError(
            f"Count must be a positive integer, got {count}"
        )

    encoded = ""
    remaining = count

    # Consume compatibility-only 300 blocks.
    while remaining >= 300:

        encoded += "?"

        remaining -= 300

    if remaining == 0:

        return encoded

    # For 251-299, use the authentic 200 glyph plus the remainder.
    if remaining > 250:

        encoded += COUNT_ENCODING[
            200
        ]

        remaining -= 200

    if remaining in COUNT_ENCODING:

        return (
            encoded
            +
            COUNT_ENCODING[
                remaining
            ]
        )

    tens = (
        remaining
        //
        10
    ) * 10

    ones = (
        remaining
        %
        10
    )

    return (
        encoded
        +
        COUNT_ENCODING[
            tens
        ]
        +
        COUNT_ENCODING[
            ones
        ]
    )


def get_count_encoding_boundaries() -> List[int]:
    """
    Return useful authentic count-boundary cases for testing.

    Includes:

        1-9
        decade transitions
        maximum authentic count

    All returned values are within 1-259.
    """

    boundaries: Set[int] = set()

    # Individual unit values.
    boundaries.update(
        range(
            1,
            10,
        )
    )

    # Decade boundaries:
    #
    # 9 / 10
    # 19 / 20
    # ...
    # 249 / 250

    for tens in range(
        10,
        MAX_AUTHENTIC_COUNT + 1,
        10,
    ):

        previous = (
            tens
            -
            1
        )

        if (
            MIN_AUTHENTIC_COUNT
            <= previous
            <= MAX_AUTHENTIC_COUNT
        ):

            boundaries.add(
                previous
            )

        if (
            MIN_AUTHENTIC_COUNT
            <= tens
            <= MAX_AUTHENTIC_COUNT
        ):

            boundaries.add(
                tens
            )

    boundaries.add(
        MAX_AUTHENTIC_COUNT
    )

    return sorted(
        boundaries
    )


# ============================================================================
# SYMBOL API
# ============================================================================

def is_valid_symbol(
    symbol: str,
) -> bool:
    """
    Return True only for symbols with an unambiguous source-defined
    UP/DOWN direction.

    Ambiguous symbols are deliberately excluded.
    """

    return (
        symbol
        in TRUSTED_SYMBOLS
    )


def validate_symbol(
    symbol: str,
) -> None:
    """
    Raise ValueError if the symbol is not in the trusted Talamat symbol set.
    """

    if not is_valid_symbol(
        symbol
    ):

        raise ValueError(
            f"Symbol {symbol!r} is not in the trusted Talamat symbol set. "
            f"UP-only: {EXCLUSIVE_UP_SYMBOLS}. "
            f"DOWN-only: {EXCLUSIVE_DOWN_SYMBOLS}. "
            f"Ambiguous/excluded: {sorted(AMBIGUOUS_SYMBOLS)}."
        )


def is_valid_symbol_direction(
    symbol: str,
    direction: str,
) -> bool:
    """
    Check whether a symbol/direction pair agrees with the original
    Talamat symbol lists.

    Ambiguous symbols intentionally return False.
    """

    if direction not in VALID_DIRECTIONS:

        return False

    if symbol in AMBIGUOUS_SYMBOLS:

        return False

    if direction == "UP":

        return (
            symbol
            in EXCLUSIVE_UP_SYMBOLS
        )

    return (
        symbol
        in EXCLUSIVE_DOWN_SYMBOLS
    )


def symbol_direction(
    symbol: str,
) -> str:
    """
    Return the trusted Talamat direction for a symbol.

    Returns:
        "UP"
        or
        "DOWN"

    Ambiguous / unsupported symbols raise ValueError.
    """

    validate_symbol(
        symbol
    )

    if symbol in EXCLUSIVE_UP_SYMBOLS:

        return "UP"

    return "DOWN"


# ============================================================================
# CANONICAL THREE-LINE TALIM REPRESENTATION
# ============================================================================

def canonical_row(
    runs,
):
    """
    Reconstruct one logical Talim row using the rendering logic found in
    the original ImageToTalam implementation.

    Each run is expected to contain:

        {
            "symbol": ...,
            "count": ...
        }

    Output:

        (
            upper_line,
            count_line,
            lower_line,
        )

    IMPORTANT — PENDING DOMAIN CONFIRMATION
    ---------------------------------------
    Multi-character count encodings currently occupy multiple horizontal
    character positions because the original implementation effectively
    uses:

        width = max(len(symbol), len(encoded_count))

    Therefore, a count encoded using multiple glyphs can shift the starting
    position of the following Talim run.

    This reproduces the provided Talamat implementation.

    It does NOT independently prove that this is the authentic historical
    Talim convention. The alignment rule can be updated here later if the
    domain expert confirms a different convention.
    """

    upper = []
    center = []
    lower = []

    for run in runs:

        symbol = run[
            "symbol"
        ]

        count = run[
            "count"
        ]

        direction = symbol_direction(
            symbol
        )

        encoded = encode_count(
            count
        )

        width = max(
            len(
                symbol
            ),
            len(
                encoded
            ),
        )

        if direction == "UP":

            upper.append(
                symbol.ljust(
                    width
                )
            )

            lower.append(
                " " * width
            )

        else:

            upper.append(
                " " * width
            )

            lower.append(
                symbol.ljust(
                    width
                )
            )

        center.append(
            encoded.rjust(
                width
            )
        )

    return tuple(
        "".join(
            parts
        ).rstrip()
        for parts in (
            upper,
            center,
            lower,
        )
    )


def canonical_text(
    sample,
) -> str:
    """
    Convert all logical rows of a structured sample into canonical
    three-line Talim text.

    A blank line separates logical rows.
    """

    return "".join(
        (
            "\n".join(
                canonical_row(
                    row[
                        "runs"
                    ]
                )
            )
            +
            "\n\n"
        )
        for row in sample[
            "logical_rows"
        ]
    )


# ============================================================================
# SEMANTIC IDENTITY
# ============================================================================

def semantic_key(
    sample,
) -> str:
    """
    Return a deterministic semantic hash for a Talim sample.

    The key ignores:

        filenames
        random seeds
        image-rendering details
        redundant metadata

    It preserves:

        row boundaries
        symbol
        count
        trusted UP/DOWN direction
    """

    rows = [
        [
            (
                run[
                    "symbol"
                ],
                run[
                    "count"
                ],
                symbol_direction(
                    run[
                        "symbol"
                    ]
                ),
            )
            for run in row[
                "runs"
            ]
        ]
        for row in sample[
            "logical_rows"
        ]
    ]

    payload = json.dumps(
        rows,
        separators=(
            ",",
            ":",
        ),
    ).encode(
        "utf-8"
    )

    return hashlib.sha256(
        payload
    ).hexdigest()