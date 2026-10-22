"""Contracts shared by Ayesha's row APIs and the root Stage-1 tier pipeline."""
import hashlib
import json
from pathlib import Path

import pytest
import torch

from data.vocabulary import (
    TalimVocabulary, build_vocabulary_from_labels, build_tier_vocabulary_from_labels,
)
from data.encoder import CTCLabelEncoder
from data.ctc_decode import ctc_collapse, collapse_ctc_sequence, decode_path, greedy_ctc_decode
from training.manifest_training import load_vocabulary
from preprocessing import normalize, preprocess_image

ROOT = Path(__file__).resolve().parents[1]


def test_row_defaults_and_explicit_tier_mapping_coexist():
    row = TalimVocabulary()
    assert len(row) == 67 and row.blank_token == "<blank>"
    assert "\n" in row.char_to_index
    supplied = TalimVocabulary(["B", "A"])
    assert supplied.char_to_index == {"B": 1, "A": 2}
    tier = TalimVocabulary.for_tiers(["B", "A", "B"])
    assert tier.char_to_index == {"A": 1, "B": 2}
    assert tier.blank_index == 0 and tier.blank_token == "<BLANK>"
    with pytest.raises(ValueError):
        TalimVocabulary.for_tiers(["A", "<BLANK>"])
    text = "ABBA"
    assert CTCLabelEncoder(tier).decode_tokens(tier.encode(text)) == text
    assert callable(normalize) and callable(preprocess_image)


def test_label_builders_keep_their_distinct_contracts(tmp_path):
    with (tmp_path / "0.txt").open("w", encoding="utf-8", newline="") as stream:
        stream.write(" B A\r\n")
    row = build_vocabulary_from_labels(tmp_path)
    tier = build_tier_vocabulary_from_labels(tmp_path)
    assert "\n" in row.char_to_index
    assert list(tier.char_to_index) == [" ", "A", "B"]
    assert tier.decode(tier.encode(" B A")) == " B A"
    with pytest.raises(FileNotFoundError):
        build_tier_vocabulary_from_labels(tmp_path / "missing")
    assert len(build_vocabulary_from_labels(tmp_path / "missing")) == 67
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError):
        build_tier_vocabulary_from_labels(empty)


def test_fixed_checkpoint_alphabet_and_both_ctc_decoders():
    mapping = json.loads((ROOT / "config/vocabulary.json").read_text())["char_to_idx"]
    tier, _ = load_vocabulary(ROOT / "config/vocabulary.json")
    assert tier.char_to_index == mapping and len(tier) == 66
    assert tier.index_to_char[0] == "<BLANK>"
    vocab = TalimVocabulary.for_tiers(["A"])
    frames = torch.tensor([0, 1, 1, 0, 1])
    assert collapse_ctc_sequence(frames) == ctc_collapse(frames) == [1, 1]
    assert greedy_ctc_decode(frames, vocab) == decode_path(frames, vocab) == "AA"
    assert decode_path([999], vocab) == ""
    with pytest.raises(KeyError):
        greedy_ctc_decode([999], vocab)


@pytest.mark.parametrize("path,expected", [["results/.gitattributes","5d51fdbec1e816cfeee1b49376b36ab1b458842b"],["results/stage1/artifact_integrity.json","75504911497d3c6a3829bd229a5c5e303e4b65b0"],["results/stage1/final_evaluation_results.json","3e1b610e0b48dea0db7fca38509c5907874be9e8"],["results/stage1/final_evaluation_summary.csv","5ba23b32c9da596cbe38b7524de5baa35d9d0942"],["results/stage1/metrics.csv","8dad97e033a69cd7009c9627fe5d01d4951e2ed0"],["results/stage1/numerical_failures.jsonl","6e9f90b36e991e9e9b9401faee9d5e8ae0ab94ca"]])
def test_saved_results_keep_original_git_blob_bytes(path, expected):
    content = (ROOT / path).read_bytes()
    identity = b"blob " + str(len(content)).encode("ascii") + b"\0" + content
    assert hashlib.sha1(identity).hexdigest() == expected


def test_repository_has_one_root_implementation():
    assert not (ROOT / "stage1").exists()
    for name in ("config", "data", "preprocessing", "talim", "models", "training",
                 "evaluation", "tests", "colab", "results", ".github/workflows"):
        assert (ROOT / name).is_dir()
