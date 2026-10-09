"""
data.ctc_decode
===============
CTC *decoding support* for NAQSHKASH (Day 5 - Ayesha Amer). The greedy decoder over model
outputs (Day 6) builds on the functions in this module.

Contents
--------
* `ctc_collapse`       raw frame-level path  ->  token ids (merge repeats, drop blanks)
* `ids_to_text`        token ids -> text (via TalimVocabulary)
* `parse_tier_text`    decoded "upper\\ncount\\nlower" text -> `DecodedTiers`
* `parse_run_text`     decoded run text ("TsFf...") -> list of `DecodedRun`
* `decode_path`        frame path -> text in one call

All functions are pure python / vocabulary based and work for any dataset using the Talim
vocabulary. Predictions can be arbitrary garbage, so nothing here raises on malformed
input: problems are reported in the returned objects (``valid`` / ``issues``).
"""

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Set, Union

from .vocabulary import (
    CANONICAL_COUNT_GLYPHS,
    CANONICAL_LOWER_SYMBOLS,
    CANONICAL_UPPER_SYMBOLS,
    TalimVocabulary,
)


def _to_list(ids) -> List[int]:
    if hasattr(ids, "detach"):  # torch tensor
        ids = ids.detach().cpu().tolist()
    elif hasattr(ids, "tolist"):  # numpy array
        ids = ids.tolist()
    return [int(i) for i in ids]


def ctc_collapse(path: Union[Sequence[int], "object"], blank_index: int = 0) -> List[int]:
    """
    Standard CTC collapse of a frame-level path: merge consecutive repeats, then drop blanks.

    >>> ctc_collapse([0, 5, 5, 0, 5, 7, 7, 0])
    [5, 5, 7]

    The blank between two identical labels is what keeps "jj" from collapsing to "j".
    """
    out: List[int] = []
    prev: Optional[int] = None
    for t in _to_list(path):
        if t != prev and t != blank_index:
            out.append(t)
        prev = t
    return out


def ids_to_text(ids: Sequence[int], vocabulary: TalimVocabulary) -> str:
    """Token ids -> text. Blanks are skipped; ids outside the vocabulary are ignored."""
    chars: List[str] = []
    for i in _to_list(ids):
        if i == vocabulary.blank_index:
            continue
        c = vocabulary.index_to_char.get(i)
        if c is not None:
            chars.append(c)
    return "".join(chars)


def decode_path(path: Sequence[int], vocabulary: TalimVocabulary) -> str:
    """Frame-level path (e.g. argmax over classes per time step) -> text."""
    return ids_to_text(ctc_collapse(path, vocabulary.blank_index), vocabulary)


# ----------------------------------------------------------------------------------------
# Structured interpretation of decoded text
# ----------------------------------------------------------------------------------------

_UP: Set[str] = set(CANONICAL_UPPER_SYMBOLS)
_DOWN: Set[str] = set(CANONICAL_LOWER_SYMBOLS)
_COUNT: Set[str] = set(CANONICAL_COUNT_GLYPHS)


def token_kind(ch: str) -> str:
    """'up', 'down', 'count', 'space', 'newline' or 'other' for a decoded character."""
    if ch in _UP:
        return "up"
    if ch in _DOWN:
        return "down"
    if ch in _COUNT:
        return "count"
    if ch == " ":
        return "space"
    if ch == "\n":
        return "newline"
    return "other"


@dataclass
class DecodedTiers:
    """Result of splitting a decoded row text into its three tiers."""
    upper: str
    count: str
    lower: str
    valid: bool
    issues: List[str] = field(default_factory=list)

    def text(self) -> str:
        return f"{self.upper}\n{self.count}\n{self.lower}"


def parse_tier_text(text: str) -> DecodedTiers:
    """
    Split a decoded row ("upper\\ncount\\nlower") into tiers and check tier content.

    Valid means: exactly 3 lines, the upper tier contains only UP symbols (and spaces), the
    count tier only count glyphs (and spaces), the lower tier only DOWN symbols (and spaces).
    Wrong line counts are repaired (missing lines -> empty, extra lines are joined into the
    lower tier) so downstream code always gets three strings, but ``valid`` is False.
    """
    issues: List[str] = []
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if len(lines) != 3:
        issues.append(f"expected 3 lines, got {len(lines)}")
        while len(lines) < 3:
            lines.append("")
        if len(lines) > 3:
            lines = [lines[0], lines[1], "".join(lines[2:])]
    upper, count, lower = lines
    for name, tier, allowed in (("upper", upper, _UP), ("count", count, _COUNT), ("lower", lower, _DOWN)):
        bad = sorted({c for c in tier if c != " " and c not in allowed})
        if bad:
            issues.append(f"{name} tier contains {bad!r}")
    if not count.strip():
        issues.append("count tier is empty")
    return DecodedTiers(upper=upper, count=count, lower=lower, valid=not issues, issues=issues)


@dataclass
class DecodedRun:
    symbol: str
    encoded_count: str
    direction: str  # 'UP' | 'DOWN'


@dataclass
class DecodedRuns:
    runs: List[DecodedRun]
    valid: bool
    issues: List[str] = field(default_factory=list)


def parse_run_text(text: str) -> DecodedRuns:
    """
    Parse decoded "runs" text: ``<symbol><count glyph(s)>`` repeated, e.g. "TsFfJkb".
    A valid run is one symbol followed by 1-2 count glyphs.
    """
    runs: List[DecodedRun] = []
    issues: List[str] = []
    cur_sym: Optional[str] = None
    cur_cnt = ""

    def flush():
        nonlocal cur_sym, cur_cnt
        if cur_sym is None:
            return
        if not 1 <= len(cur_cnt) <= 2:
            issues.append(f"run {cur_sym!r} has {len(cur_cnt)} count glyphs (expected 1-2)")
        runs.append(DecodedRun(cur_sym, cur_cnt, "UP" if cur_sym in _UP else "DOWN"))
        cur_sym, cur_cnt = None, ""

    for ch in text:
        kind = token_kind(ch)
        if kind in ("up", "down"):
            flush()
            cur_sym = ch
        elif kind == "count":
            if cur_sym is None:
                issues.append(f"count glyph {ch!r} before any symbol")
            else:
                cur_cnt += ch
        else:
            issues.append(f"unexpected character {ch!r} in run text")
    flush()
    return DecodedRuns(runs=runs, valid=not issues and bool(runs), issues=issues)
