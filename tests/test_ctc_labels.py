"""
tests.test_ctc_labels
=====================
Day 5 tests (Ayesha Amer): CTC label parsing / encoding / decoding support.

All tests build their own tiny datasets in a temp folder, so they run on any machine. Set the
environment variable ``NAQSHKASH_DATASET_DIR`` to a real dataset (split or dataset folder) to also
run the real-data tests at the bottom.
"""
import json
import os
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from data.ctc_decode import (
    ctc_collapse,
    decode_path,
    ids_to_text,
    parse_run_text,
    parse_tier_text,
    token_kind,
)
from data.ctc_labels import (
    CTCLengthError,
    LabelFormatError,
    RowLabel,
    Run,
    build_ctc_batch,
    collate_ctc_targets,
    discover_samples,
    find_ctc_problems,
    input_lengths_from_widths,
    load_row_labels,
    parse_label_text,
    required_ctc_length,
    row_texts,
    validate_ctc_batch,
)
from data.encoder import CTCLabelEncoder
from data.vocabulary import (
    CANONICAL_COUNT_GLYPHS,
    CANONICAL_LOWER_SYMBOLS,
    CANONICAL_UPPER_SYMBOLS,
    TalimVocabulary,
)

TWO_ROWS = "    N\njhjic\nY]R\n\n BGCJ NM\nafbajfajc\nX\n\n"
EMPTY_LOWER = " P\naj\n\n\n"            # row without lower tier
EMPTY_UPPER = "\naj\nW\n\n"             # row without upper tier


@pytest.fixture(scope="module")
def vocab():
    return TalimVocabulary()


# ----------------------------------------------------------------------------------------
# Parsing
# ----------------------------------------------------------------------------------------

def test_parse_two_rows():
    rows = parse_label_text(TWO_ROWS)
    assert len(rows) == 2
    assert (rows[0].upper, rows[0].count, rows[0].lower) == ("    N", "jhjic", "Y]R")
    assert (rows[1].upper, rows[1].count, rows[1].lower) == (" BGCJ NM", "afbajfajc", "X")


def test_parse_keeps_empty_tiers_not_split_on_blank_lines():
    """Empty upper/lower tiers create '\\n\\n' INSIDE a row; rows must still be 4-line groups."""
    rows = parse_label_text(EMPTY_LOWER + EMPTY_UPPER)
    assert len(rows) == 2
    assert (rows[0].upper, rows[0].count, rows[0].lower) == (" P", "aj", "")
    assert (rows[1].upper, rows[1].count, rows[1].lower) == ("", "aj", "W")


def test_parse_tolerates_crlf_missing_final_newline_and_trailing_spaces():
    text = " P  \r\naj\r\nW \r\n\r\n"
    assert parse_label_text(text)[0].upper == " P"
    assert len(parse_label_text(" P\naj\nW")) == 1          # no separator, no final newline
    assert len(parse_label_text(" P\naj\nW\n")) == 1        # no separator line
    assert parse_label_text("") == [] and parse_label_text("\n\n") == []


def test_parse_keeps_leading_spaces():
    assert parse_label_text("   N\nj\nR\n\n")[0].upper == "   N"


def test_parse_rejects_bad_layouts():
    with pytest.raises(LabelFormatError):
        parse_label_text("A\nB\n")                  # 2 lines
    with pytest.raises(LabelFormatError):
        parse_label_text("A\nB\nC\nD\n")            # separator line has text


def test_row_text_modes():
    r = RowLabel("  N", "jh", "Y", runs=(Run("Y", "j"), Run("N", "h")))
    assert r.text("tiers") == "  N\njh\nY"
    assert r.text("runs") == "YjNh"
    with pytest.raises(LabelFormatError):
        RowLabel("A", "a", "").text("runs")
    with pytest.raises(ValueError):
        r.text("nope")


# ----------------------------------------------------------------------------------------
# Dataset discovery + JSON agreement
# ----------------------------------------------------------------------------------------

def _write_sample(folder: Path, sid: str, rows, with_json=True, with_image=False):
    (folder / "labels").mkdir(parents=True, exist_ok=True)
    (folder / "labels" / f"{sid}.txt").write_text(
        "".join(f"{u}\n{c}\n{l}\n\n" for u, c, l, _ in rows), encoding="utf-8")
    if with_json:
        (folder / "json").mkdir(exist_ok=True)
        meta = {"logical_rows": [
            {"row_index": i, "upper": u, "count_line": c, "lower": l,
             "runs": [{"symbol": s, "count": 1, "direction": "UP", "encoded_count": e} for s, e in runs]}
            for i, (u, c, l, runs) in enumerate(rows)]}
        (folder / "json" / f"{sid}.json").write_text(json.dumps(meta), encoding="utf-8")
    if with_image:
        (folder / "images").mkdir(exist_ok=True)
        (folder / "images" / f"{sid}.png").write_bytes(b"x")


ROWS = [(" P", "aj", "W", [("W", "a"), ("P", "j")]), ("", "f", "X", [("X", "f")])]


def test_discover_split_and_dataset_layouts(tmp_path):
    _write_sample(tmp_path / "ds" / "train", "t1", ROWS, with_image=True)
    _write_sample(tmp_path / "ds" / "validation", "v1", ROWS)
    found = discover_samples(tmp_path / "ds")
    assert [(s.split, s.sample_id) for s in found] == [("train", "t1"), ("validation", "v1")]
    assert found[0].image_path is not None and found[1].image_path is None
    assert len(discover_samples(tmp_path / "ds" / "train")) == 1


def test_discover_flat_layout(tmp_path):
    (tmp_path / "a.txt").write_text(" P\naj\nW\n\n", encoding="utf-8")
    (tmp_path / "a.png").write_bytes(b"x")
    s = discover_samples(tmp_path)
    assert len(s) == 1 and s[0].image_path is not None and s[0].json_path is None
    with pytest.raises(FileNotFoundError):
        discover_samples(tmp_path / "missing")


def test_load_row_labels_attaches_runs_and_checks_json(tmp_path):
    _write_sample(tmp_path, "s", ROWS)
    rows = load_row_labels(tmp_path / "labels" / "s.txt", tmp_path / "json" / "s.json", need_runs=True)
    assert row_texts(rows, "runs") == ["WaPj", "Xf"]
    assert row_texts(rows, "tiers") == [" P\naj\nW", "\nf\nX"]
    # TXT/JSON disagree -> error
    meta = json.loads((tmp_path / "json" / "s.json").read_text())
    meta["logical_rows"][0]["count_line"] = "zz"
    (tmp_path / "json" / "s.json").write_text(json.dumps(meta))
    with pytest.raises(LabelFormatError):
        load_row_labels(tmp_path / "labels" / "s.txt", tmp_path / "json" / "s.json")


def test_runs_mode_without_json_raises(tmp_path):
    _write_sample(tmp_path, "s", ROWS, with_json=False)
    with pytest.raises(LabelFormatError):
        load_row_labels(tmp_path / "labels" / "s.txt", None, need_runs=True)


# ----------------------------------------------------------------------------------------
# Token mapping
# ----------------------------------------------------------------------------------------

def test_vocab_json_matches_canonical_and_is_bijective(vocab):
    repo_json = Path(__file__).resolve().parent.parent / "data" / "vocab.json"
    if repo_json.exists():
        assert TalimVocabulary.load(repo_json).char_to_index == vocab.char_to_index
    assert len(vocab) == 67 and vocab.blank_index == 0
    assert sorted(vocab.char_to_index.values()) == list(range(1, 67))
    for ch, i in vocab.char_to_index.items():
        assert vocab.index_to_char[i] == ch


def test_symbol_and_count_sets_are_disjoint_and_complete():
    up, down, cnt = set(CANONICAL_UPPER_SYMBOLS), set(CANONICAL_LOWER_SYMBOLS), set(CANONICAL_COUNT_GLYPHS)
    assert (len(up), len(down), len(cnt)) == (16, 14, 34)
    assert not (up & down) and not (up & cnt) and not (down & cnt)
    assert {token_kind(c) for c in up} == {"up"} and {token_kind(c) for c in down} == {"down"}
    assert {token_kind(c) for c in cnt} == {"count"}
    assert token_kind(" ") == "space" and token_kind("\n") == "newline" and token_kind("?") == "other"


def test_every_token_round_trips_individually(vocab):
    for ch in vocab.characters:
        assert vocab.decode(vocab.encode(ch)) == ch


def test_row_label_round_trip_through_ctc_path(vocab):
    for text in (" P\naj\nW", "\naj\nW", " P\naj\n", "  N\njhjic\nY]R", "jj\n\n"):
        ids = vocab.encode(text)
        # frame path with repeats and blanks, like a real network output
        path = []
        for t in ids:
            path += [0, t, t]
        path += [0]
        assert decode_path(path, vocab) == text


# ----------------------------------------------------------------------------------------
# CTC collapse / decode
# ----------------------------------------------------------------------------------------

def test_ctc_collapse_rules():
    assert ctc_collapse([0, 5, 5, 0, 5, 7, 7, 0]) == [5, 5, 7]
    assert ctc_collapse([5, 5, 5]) == [5]
    assert ctc_collapse([0, 0, 0]) == []
    assert ctc_collapse([]) == []
    assert ctc_collapse(torch.tensor([0, 3, 3, 4])) == [3, 4]
    assert ctc_collapse([9, 0, 9], blank_index=0) == [9, 9]
    assert ctc_collapse([1, 2, 2, 0, 3], blank_index=2) == [1, 0, 3]


def test_ids_to_text_ignores_blank_and_unknown_ids(vocab):
    ids = vocab.encode("jj")
    assert ids_to_text([0] + ids + [999, -4], vocab) == "jj"
    assert ids_to_text(torch.tensor(ids), vocab) == "jj"


# ----------------------------------------------------------------------------------------
# Structured decoding
# ----------------------------------------------------------------------------------------

def test_parse_tier_text_valid_and_invalid():
    ok = parse_tier_text("  N\njhjic\nY]R")
    assert ok.valid and (ok.upper, ok.count, ok.lower) == ("  N", "jhjic", "Y]R")
    assert parse_tier_text("\naj\n").valid                      # empty tiers are fine
    swapped = parse_tier_text("aj\nN\nR")
    assert not swapped.valid and any("upper" in i for i in swapped.issues)
    two = parse_tier_text("N\nj")
    assert not two.valid and (two.upper, two.count, two.lower) == ("N", "j", "")
    four = parse_tier_text("N\nj\nR\nR")
    assert not four.valid and four.lower == "RR"
    assert not parse_tier_text("N\n\nR").valid                  # empty count tier


def test_parse_run_text():
    r = parse_run_text("TsFfJkb")
    assert r.valid
    assert [(x.symbol, x.encoded_count, x.direction) for x in r.runs] == [
        ("T", "s", "DOWN"), ("F", "f", "UP"), ("J", "kb", "UP")]
    assert not parse_run_text("sT").valid                       # count before symbol
    assert not parse_run_text("TT").valid                       # run without count
    assert not parse_run_text("Tabc").valid                     # >2 count glyphs
    assert not parse_run_text("").valid


# ----------------------------------------------------------------------------------------
# CTC length rules
# ----------------------------------------------------------------------------------------

def test_required_ctc_length_counts_repeats():
    assert required_ctc_length([]) == 0
    assert required_ctc_length([1, 2, 3]) == 3
    assert required_ctc_length([1, 1]) == 3
    assert required_ctc_length([1, 1, 1]) == 5
    assert required_ctc_length([2, 1, 1, 2]) == 5


def test_required_length_matches_torch_ctc_feasibility():
    """T == required length gives finite loss; T == required-1 is impossible (inf)."""
    ids = [4, 4, 7, 7, 7]
    need = required_ctc_length(ids)
    ctc = nn.CTCLoss(blank=0, reduction="sum")
    tgt = torch.tensor(ids)
    for T, finite in ((need, True), (need - 1, False)):
        lp = torch.randn(T, 1, 10).log_softmax(2)
        loss = ctc(lp, tgt, torch.tensor([T]), torch.tensor([len(ids)]))
        assert bool(torch.isfinite(loss)) == finite


def test_input_lengths_from_widths():
    assert input_lengths_from_widths([100, 103, 3], 4) == [25, 25, 1]
    assert input_lengths_from_widths([103], 4, rounding="ceil") == [26]
    assert input_lengths_from_widths([100], 4, offset=1) == [26]
    with pytest.raises(ValueError):
        input_lengths_from_widths([10], 0)
    with pytest.raises(ValueError):
        input_lengths_from_widths([10], 4, rounding="round")


def test_validate_ctc_batch_reports_all_problems():
    ids = [[1, 2, 3], [4, 4, 4, 4], [], [0, 5], [70]]
    lens = [10, 5, 10, 10, 10]
    probs = find_ctc_problems(lens, ids, blank_index=0, vocab_size=67)
    reasons = {p.index: p.reason for p in probs}
    assert set(reasons) == {1, 2, 3, 4}
    assert "needs 7" in reasons[1] and "empty" in reasons[2]
    with pytest.raises(CTCLengthError) as e:
        validate_ctc_batch(lens, ids, vocab_size=67)
    assert "4 sample(s)" in str(e.value)
    assert validate_ctc_batch(lens, ids, vocab_size=67, raise_error=False)
    assert validate_ctc_batch([10], [[1, 2]]) == []
    with pytest.raises(ValueError):
        validate_ctc_batch([1], [[1], [2]])


# ----------------------------------------------------------------------------------------
# Tensors for CTCLoss
# ----------------------------------------------------------------------------------------

def test_collate_ctc_targets(vocab):
    texts = ["  N\njhjic\nY]R", " P\naj\nW", "j"]
    b = collate_ctc_targets(texts, vocab)
    lens = [len(t) for t in texts]
    assert b["target_lengths"].tolist() == lens
    assert b["targets"].shape == (sum(lens),) and b["targets"].dtype == torch.long
    assert b["targets_padded"].shape == (3, max(lens))
    flat = b["targets"].tolist()
    pos = 0
    for i, t in enumerate(texts):
        seg = flat[pos:pos + lens[i]]
        assert seg == vocab.encode(t) == b["targets_padded"][i, :lens[i]].tolist() == b["token_ids"][i]
        pos += lens[i]
    assert (b["targets"] > 0).all()                      # blank (0) never in a target
    # identical result to the Day-4 encoder
    ref = CTCLabelEncoder(vocab).encode_batch(texts)
    assert torch.equal(ref["targets"], b["targets"])
    assert torch.equal(ref["target_lengths"], b["target_lengths"])
    with pytest.raises(TypeError):
        collate_ctc_targets("abc", vocab)
    with pytest.raises(KeyError):
        collate_ctc_targets(["A\n?\nB"], vocab)


def test_build_ctc_batch_validates(vocab):
    b = build_ctc_batch([" P\naj\nW"], [100], vocab, downsample=4)
    assert b["input_lengths"].tolist() == [25]
    with pytest.raises(CTCLengthError):
        build_ctc_batch([" P\naj\nW"], [20], vocab, downsample=4)    # 5 steps < 8 needed
    assert build_ctc_batch([" P\naj\nW"], [20], vocab, downsample=4, validate=False)


class _TinyCRNN(nn.Module):
    """Stand-in for Hareem's CRNN (CNN stride 4 horizontally + BiLSTM + linear); test-only."""

    def __init__(self, n_classes=67):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.AdaptiveAvgPool2d((1, None)),
        )
        self.rnn = nn.LSTM(16, 16, bidirectional=True)
        self.fc = nn.Linear(32, n_classes)

    def forward(self, x):                       # x (N,1,64,W) -> (T,N,C)
        f = self.cnn(x).squeeze(2).permute(2, 0, 1)
        return self.fc(self.rnn(f)[0]).log_softmax(2)


def test_batch_flows_through_model_into_ctc_loss(vocab):
    """Labels + widths -> input lengths -> CTCLoss on a model output is finite and backprops."""
    widths = [96, 160, 72]
    texts = ["  N\njhjic\nY", " P\naj\nW", "A\nj\nR"]
    imgs = torch.ones(3, 1, 64, max(widths))                    # padded with white (1.0)
    model = _TinyCRNN()
    out = model(imgs)
    assert out.shape[0] == max(widths) // 4                     # stride-4 time axis
    batch = build_ctc_batch(texts, widths, vocab, downsample=4)
    loss = nn.CTCLoss(blank=vocab.blank_index, zero_infinity=False)(
        out, batch["targets"], batch["input_lengths"], batch["target_lengths"])
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None for p in model.parameters())
    # padded-target form gives the same loss
    loss2 = nn.CTCLoss(blank=0)(out, batch["targets_padded"], batch["input_lengths"], batch["target_lengths"])
    assert torch.allclose(loss, loss2)


# ----------------------------------------------------------------------------------------
# Optional: real dataset (set NAQSHKASH_DATASET_DIR)
# ----------------------------------------------------------------------------------------

REAL = os.environ.get("NAQSHKASH_DATASET_DIR")
REAL = REAL if REAL and Path(REAL).exists() else None   # unset/missing folder -> tests are skipped


@pytest.mark.skipif(not REAL, reason="set NAQSHKASH_DATASET_DIR to run real-dataset tests")
def test_real_dataset_labels_roundtrip_both_modes(vocab):
    samples = discover_samples(REAL)
    assert samples, "no samples found"
    for s in samples:
        rows = load_row_labels(s.label_path, s.json_path)
        for r in rows:
            for mode in ("tiers", "runs") if r.runs else ("tiers",):
                text = r.text(mode)
                assert vocab.decode(vocab.encode(text)) == text
                assert (parse_tier_text(text) if mode == "tiers" else parse_run_text(text)).valid


@pytest.mark.skipif(not REAL, reason="set NAQSHKASH_DATASET_DIR to run real-dataset tests")
def test_real_dataset_rows_match_images_and_are_ctc_feasible(vocab):
    np = pytest.importorskip("numpy")
    Image = pytest.importorskip("PIL.Image")
    from preprocessing.model_prep import preprocess_image

    checked = 0
    for s in discover_samples(REAL)[:60]:
        if s.image_path is None:
            continue
        rows = load_row_labels(s.label_path, s.json_path)
        res = preprocess_image(np.array(Image.open(s.image_path)))
        assert res.num_rows == len(rows), s.sample_id
        widths = [r.tensor_image.shape[1] for r in res.rows]
        validate_ctc_batch(
            input_lengths_from_widths(widths, 4),
            [vocab.encode(r.text("tiers")) for r in rows],
            vocab_size=len(vocab),
        )
        checked += 1
    assert checked > 0