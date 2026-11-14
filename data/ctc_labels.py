"""
data.ctc_labels
===============
Dataset-agnostic CTC *label* support for NAQSHKASH (Day 5 - Ayesha Amer).

What this module does
---------------------
1. Reads label files of ANY Talim dataset laid out as ``<split>/labels/*.txt`` (optionally with
   ``<split>/json/*.json`` and ``<split>/images/*.png`` next to it) and turns every sample into
   one label string **per logical row** (the unit that preprocessing / the CRNN sees).
2. Two label modes are supported (same vocabulary, same encoder):
     * ``"tiers"`` (default, the project convention): ``"<upper>\\n<count>\\n<lower>"``
     * ``"runs"``  (optional): left-to-right ``<symbol><count glyphs>`` per run, e.g. ``"TsFfJkb"``.
       Needs the JSON ``logical_rows[*].runs`` list.
3. Builds the tensors ``nn.CTCLoss`` needs (flat or padded targets, target lengths), computes
   CTC input lengths from image widths and VALIDATES a batch before the loss is computed
   (a label that is longer than the number of output time steps gives ``inf`` loss / NaN).

Label-file contract (what the parser accepts)
---------------------------------------------
Each logical row is 4 lines: ``upper``, ``count``, ``lower`` and one empty separator line::

    "    N\\njhjic\\nY]R\\n\\n BGCJ NM\\nafbajfajc\\nX\\n\\n"

Empty tiers are legal (an empty upper/lower tier is just an empty line). Therefore rows are
NOT split on blank lines (that breaks on rows with an empty tier) but on groups of 4 lines.
Windows line endings, a missing final newline and trailing spaces are tolerated.

Only pure-python helpers are used for parsing (no torch import needed); torch is imported lazily
by the tensor helpers.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union
import json

from .vocabulary import TalimVocabulary, normalize_label_text

LABEL_MODES = ("tiers", "runs")
LINES_PER_ROW = 4  # upper, count, lower, empty separator


class LabelFormatError(ValueError):
    """Raised when a label file does not follow the expected row layout."""


class CTCLengthError(ValueError):
    """Raised when a batch cannot be trained with CTC (target longer than output sequence)."""


# ----------------------------------------------------------------------------------------
# Row label dataclasses
# ----------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Run:
    """One run of a logical row (symbol + its count glyph(s))."""
    symbol: str
    encoded_count: str
    count: Optional[int] = None
    direction: Optional[str] = None


@dataclass(frozen=True)
class RowLabel:
    """Ground truth of one logical Talim row."""
    upper: str
    count: str
    lower: str
    runs: Optional[Tuple[Run, ...]] = None

    def text(self, mode: str = "tiers") -> str:
        """Return the CTC target string of this row for the given label mode."""
        if mode == "tiers":
            return f"{self.upper}\n{self.count}\n{self.lower}"
        if mode == "runs":
            if not self.runs:
                raise LabelFormatError(
                    "label mode 'runs' needs run information (JSON logical_rows[*].runs); "
                    "none is available for this row."
                )
            return "".join(r.symbol + r.encoded_count for r in self.runs)
        raise ValueError(f"Unknown label mode {mode!r}; expected one of {LABEL_MODES}")


@dataclass
class SampleRef:
    """A sample found on disk (paths that do not exist are None)."""
    sample_id: str
    split: str
    label_path: Path
    json_path: Optional[Path] = None
    image_path: Optional[Path] = None


# ----------------------------------------------------------------------------------------
# Parsing
# ----------------------------------------------------------------------------------------

def _clean_line(line: str, strip_trailing: bool) -> str:
    return line.rstrip(" \t") if strip_trailing else line


def parse_label_text(
    text: str,
    lines_per_row: int = LINES_PER_ROW,
    strip_trailing: bool = True,
) -> List[RowLabel]:
    """
    Parse the content of a label file into logical rows.

    Parameters
    ----------
    text : str
        Raw label file content.
    lines_per_row : int
        Lines per row including the empty separator (4 for the NAQSHKASH datasets).
    strip_trailing : bool
        Remove trailing spaces/tabs of each line (they are invisible in the image and would only
        add un-learnable CTC targets). Leading spaces are kept: they encode the position.

    Raises
    ------
    LabelFormatError
        If the line layout does not match (e.g. a separator line contains text).
    """
    text = normalize_label_text(text)
    if text.strip() == "":
        return []
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()  # artefact of the final newline
    if len(lines) % lines_per_row == lines_per_row - 1:
        lines.append("")  # last separator line missing
    if len(lines) % lines_per_row != 0:
        raise LabelFormatError(
            f"{len(lines)} lines cannot be split into rows of {lines_per_row} lines "
            f"(upper, count, lower, empty separator)."
        )
    rows: List[RowLabel] = []
    for start in range(0, len(lines), lines_per_row):
        group = lines[start:start + lines_per_row]
        if group[lines_per_row - 1].strip() != "":
            raise LabelFormatError(
                f"Row {start // lines_per_row}: line {start + lines_per_row} should be an empty "
                f"separator but contains {group[lines_per_row - 1]!r}."
            )
        upper, count, lower = (_clean_line(g, strip_trailing) for g in group[:3])
        rows.append(RowLabel(upper=upper, count=count, lower=lower))
    return rows


def _rows_from_json(meta: Dict[str, Any], strip_trailing: bool = True) -> List[RowLabel]:
    rows = []
    for r in meta.get("logical_rows", []):
        runs = tuple(
            Run(
                symbol=x["symbol"],
                encoded_count=x["encoded_count"],
                count=x.get("count"),
                direction=x.get("direction"),
            )
            for x in r.get("runs", [])
        ) or None
        rows.append(
            RowLabel(
                upper=_clean_line(r["upper"], strip_trailing),
                count=_clean_line(r["count_line"], strip_trailing),
                lower=_clean_line(r["lower"], strip_trailing),
                runs=runs,
            )
        )
    return rows


def load_row_labels(
    label_path: Union[str, Path],
    json_path: Optional[Union[str, Path]] = None,
    need_runs: bool = False,
    strip_trailing: bool = True,
) -> List[RowLabel]:
    """
    Load the logical-row ground truth of one sample.

    The TXT file is always the reference for the tier text. If a JSON file with
    ``logical_rows`` is given, its run information is attached to the rows (and it is checked
    to agree with the TXT; a disagreement raises ``LabelFormatError``).
    """
    with open(label_path, "r", encoding="utf-8", newline="") as f:
        rows = parse_label_text(f.read(), strip_trailing=strip_trailing)

    if json_path is not None and Path(json_path).exists():
        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        jrows = _rows_from_json(meta, strip_trailing=strip_trailing)
        if jrows:
            if len(jrows) != len(rows):
                raise LabelFormatError(
                    f"{label_path}: TXT has {len(rows)} rows but JSON has {len(jrows)}."
                )
            merged = []
            for i, (t, j) in enumerate(zip(rows, jrows)):
                if (t.upper, t.count, t.lower) != (j.upper, j.count, j.lower):
                    raise LabelFormatError(f"{label_path}: row {i} differs between TXT and JSON.")
                merged.append(RowLabel(t.upper, t.count, t.lower, j.runs))
            rows = merged
    if need_runs and any(not r.runs for r in rows):
        raise LabelFormatError(f"{label_path}: run information required but not available.")
    return rows


def row_texts(rows: Sequence[RowLabel], mode: str = "tiers") -> List[str]:
    """CTC target strings (one per logical row)."""
    return [r.text(mode) for r in rows]


# ----------------------------------------------------------------------------------------
# Dataset discovery (generic)
# ----------------------------------------------------------------------------------------

_IMAGE_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _find_image(image_dir: Path, stem: str) -> Optional[Path]:
    for ext in _IMAGE_EXT:
        p = image_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def discover_samples(
    root: Union[str, Path],
    label_dir: str = "labels",
    json_dir: str = "json",
    image_dir: str = "images",
) -> List[SampleRef]:
    """
    Find all samples under ``root`` (works for any dataset that follows
    ``<folder>/labels/*.txt`` [+ ``<folder>/json/*.json``, ``<folder>/images/*.*``]).

    ``root`` may be a split folder (``train``), a dataset folder containing split folders, or a
    flat folder with ``name.txt`` / ``name.png`` pairs. The split name is the name of the folder
    that contains ``labels``.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(root)
    samples: List[SampleRef] = []

    label_folders = sorted({p.parent for p in root.rglob("*.txt") if p.parent.name == label_dir})
    if label_folders:
        for lf in label_folders:
            base = lf.parent
            for txt in sorted(lf.glob("*.txt")):
                jp = base / json_dir / f"{txt.stem}.json"
                samples.append(
                    SampleRef(
                        sample_id=txt.stem,
                        split=base.name,
                        label_path=txt,
                        json_path=jp if jp.exists() else None,
                        image_path=_find_image(base / image_dir, txt.stem),
                    )
                )
        return samples

    # flat layout: name.txt next to name.png
    for txt in sorted(root.glob("*.txt")):
        jp = root / f"{txt.stem}.json"
        samples.append(
            SampleRef(
                sample_id=txt.stem,
                split=root.name,
                label_path=txt,
                json_path=jp if jp.exists() else None,
                image_path=_find_image(root, txt.stem),
            )
        )
    return samples


# ----------------------------------------------------------------------------------------
# CTC length rules
# ----------------------------------------------------------------------------------------

def required_ctc_length(token_ids: Sequence[int]) -> int:
    """
    Minimum number of output time steps CTC needs to emit ``token_ids``:
    ``len(labels) + number of identical neighbouring labels`` (a blank must separate them,
    e.g. "jj" needs 3 steps: j, blank, j).
    """
    ids = list(token_ids)
    return len(ids) + sum(1 for a, b in zip(ids, ids[1:]) if a == b)


def input_lengths_from_widths(
    widths: Iterable[int],
    downsample: int = 4,
    rounding: str = "floor",
    offset: int = 0,
    min_length: int = 1,
) -> List[int]:
    """
    CTC input (time-step) length of each image from its pixel width.

    ``time_steps = floor(width / downsample) + offset`` (or ``ceil``). The exact value depends on
    the CRNN's pooling/stride layout: ALWAYS confirm against the real model with
    ``model(images).shape[0]`` (see `validate_ctc_batch`). ``downsample`` is the total horizontal
    stride of the CNN (e.g. 4 for two 2x2 poolings).
    """
    if downsample < 1:
        raise ValueError("downsample must be >= 1")
    if rounding not in ("floor", "ceil"):
        raise ValueError("rounding must be 'floor' or 'ceil'")
    out = []
    for w in widths:
        w = int(w)
        t = w // downsample if rounding == "floor" else -(-w // downsample)
        out.append(max(min_length, t + offset))
    return out


@dataclass
class CTCProblem:
    index: int
    reason: str


def find_ctc_problems(
    input_lengths: Sequence[int],
    targets: Sequence[Sequence[int]],
    blank_index: int = 0,
    vocab_size: Optional[int] = None,
) -> List[CTCProblem]:
    """List every sample that CTC cannot train on (pure python, no torch)."""
    problems: List[CTCProblem] = []
    if len(input_lengths) != len(targets):
        raise ValueError(
            f"input_lengths ({len(input_lengths)}) and targets ({len(targets)}) differ in length"
        )
    for i, (T, ids) in enumerate(zip(input_lengths, targets)):
        ids = list(ids)
        if len(ids) == 0:
            problems.append(CTCProblem(i, "empty target"))
            continue
        if any(t == blank_index for t in ids):
            problems.append(CTCProblem(i, "target contains the blank index"))
        if vocab_size is not None and any(t < 0 or t >= vocab_size for t in ids):
            problems.append(CTCProblem(i, f"target id outside [0, {vocab_size})"))
        need = required_ctc_length(ids)
        if T < need:
            problems.append(CTCProblem(i, f"needs {need} time steps but the model gives {T}"))
    return problems


def validate_ctc_batch(
    input_lengths: Sequence[int],
    targets: Sequence[Sequence[int]],
    blank_index: int = 0,
    vocab_size: Optional[int] = None,
    raise_error: bool = True,
) -> List[CTCProblem]:
    """Check a batch BEFORE ``nn.CTCLoss``; raises `CTCLengthError` listing the bad samples."""
    problems = find_ctc_problems(input_lengths, targets, blank_index, vocab_size)
    if problems and raise_error:
        msg = "; ".join(f"sample {p.index}: {p.reason}" for p in problems[:10])
        more = f" (+{len(problems) - 10} more)" if len(problems) > 10 else ""
        raise CTCLengthError(f"{len(problems)} sample(s) not trainable with CTC: {msg}{more}")
    return problems


# ----------------------------------------------------------------------------------------
# Tensors for nn.CTCLoss
# ----------------------------------------------------------------------------------------

def collate_ctc_targets(
    texts: Sequence[str],
    vocabulary: TalimVocabulary,
    pad_value: int = 0,
):
    """
    Encode a batch of label strings for ``nn.CTCLoss``.

    Returns a dict with
      * ``targets``         1-D LongTensor, all labels concatenated (sum(lengths),)
      * ``target_lengths``  LongTensor (N,)
      * ``targets_padded``  LongTensor (N, max_len) padded with ``pad_value`` (use with the
                            padded-target form of CTCLoss; pad_value must not matter because
                            target_lengths is always passed)
      * ``token_ids``       list of python lists (handy for `validate_ctc_batch`)
    """
    import torch

    if isinstance(texts, str):
        raise TypeError("texts must be a list of strings, not a single string")
    ids = [vocabulary.encode(t) for t in texts]
    lengths = [len(x) for x in ids]
    flat = [t for x in ids for t in x]
    max_len = max(lengths) if lengths else 0
    padded = torch.full((len(ids), max_len), int(pad_value), dtype=torch.long)
    for i, x in enumerate(ids):
        if x:
            padded[i, : len(x)] = torch.tensor(x, dtype=torch.long)
    return {
        "targets": torch.tensor(flat, dtype=torch.long),
        "target_lengths": torch.tensor(lengths, dtype=torch.long),
        "targets_padded": padded,
        "token_ids": ids,
    }


def build_ctc_batch(
    texts: Sequence[str],
    image_widths: Sequence[int],
    vocabulary: TalimVocabulary,
    downsample: int = 4,
    rounding: str = "floor",
    offset: int = 0,
    validate: bool = True,
):
    """
    One call that returns everything ``nn.CTCLoss(log_probs, targets, input_lengths,
    target_lengths)`` needs for a batch (see `collate_ctc_targets`) plus ``input_lengths``.

    ``image_widths`` are the widths of the images BEFORE padding to the batch width (padding must
    not count as time steps). If the real model's time dimension differs from the
    ``downsample`` formula, compute the lengths yourself and use `validate_ctc_batch`.
    """
    import torch

    batch = collate_ctc_targets(texts, vocabulary)
    lengths = input_lengths_from_widths(image_widths, downsample, rounding, offset)
    if validate:
        validate_ctc_batch(
            lengths, batch["token_ids"], blank_index=vocabulary.blank_index,
            vocab_size=len(vocabulary),
        )
    batch["input_lengths"] = torch.tensor(lengths, dtype=torch.long)
    return batch
