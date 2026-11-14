"""
tests.test_ayesha_vocab_encoder
===============================
Unit tests for Day 4 Talim vocabulary and CTC label encoder (`data.vocabulary`, `data.encoder`).

Author: Ayesha Amer (Day 4 Scope)
Project: NAQSHKASH FYP
"""

from pathlib import Path
import tempfile
import torch
import pytest

from data.vocabulary import (
    TalimVocabulary,
    build_vocabulary_from_labels,
    normalize_label_text,
    CANONICAL_SYMBOLS,
    CANONICAL_COUNT_GLYPHS,
)
from data.encoder import CTCLabelEncoder


def test_canonical_vocabulary_size_and_classes():
    """Verify canonical vocabulary contains 30 symbols, 34 count glyphs, space, newline, and blank at index 0."""
    vocab = TalimVocabulary()

    # 30 symbols + 34 count glyphs + ' ' + '\n' + 1 blank = 67
    assert len(vocab) == 67
    assert vocab.blank_index == 0
    assert 0 not in vocab.char_to_index.values()
    assert vocab.index_to_char[0] == "<blank>"

    # 30 symbols verified
    assert len(CANONICAL_SYMBOLS) == 30
    for sym in CANONICAL_SYMBOLS:
        assert sym in vocab.char_to_index

    # 34 count glyphs verified
    assert len(CANONICAL_COUNT_GLYPHS) == 34
    for glyph in CANONICAL_COUNT_GLYPHS:
        assert glyph in vocab.char_to_index

    # Formatting characters
    assert " " in vocab.char_to_index
    assert "\n" in vocab.char_to_index


def test_normalize_label_text():
    """Verify that all line ending conventions (\\r\\n, \\r) normalize to \\n."""
    raw = "CN\r\njjj\r\n  _\r\n\r\n N\rjjj\r] U"
    expected = "CN\njjj\n  _\n\n N\njjj\n] U"
    assert normalize_label_text(raw) == expected


def test_vocabulary_encode_decode_round_trip():
    """Verify exact round-trip encoding and decoding on real multi-tier Talim label text."""
    vocab = TalimVocabulary()
    sample_text = "CN\njjj\n  _\n\n N\njjj\n] U\n\nQE\njeif\n  R_\n\n"

    encoded = vocab.encode(sample_text)
    assert isinstance(encoded, list)
    assert all(isinstance(idx, int) and idx > 0 for idx in encoded)

    decoded = vocab.decode(encoded)
    assert decoded == sample_text


def test_unknown_character_raises_key_error():
    """Verify that unknown/unsupported characters raise informative KeyError."""
    vocab = TalimVocabulary()
    with pytest.raises(KeyError) as exc_info:
        vocab.encode("CN\njjj\n@")
    assert "@" in str(exc_info.value)


def test_ctc_label_encoder_single_label():
    """Verify that CTCLabelEncoder produces 1D torch.LongTensor with correct indices."""
    vocab = TalimVocabulary()
    encoder = CTCLabelEncoder(vocab)

    text = "CN\njjj\n  _"
    tensor = encoder.encode_text(text)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == torch.long
    assert tensor.ndim == 1
    assert tensor.shape[0] == len(text)
    assert tensor.tolist() == vocab.encode(text)


def test_ctc_label_encoder_batch_encoding():
    """Verify that batch encoding produces concatenated targets and target lengths in CTCLoss format."""
    vocab = TalimVocabulary()
    encoder = CTCLabelEncoder(vocab)

    batch_texts = [
        "CN\njjj\n  _",
        " N\njjj\n] U",
        "QE\njeif\n  R_",
    ]

    result = encoder.encode_batch(batch_texts)

    assert "targets" in result
    assert "target_lengths" in result

    targets = result["targets"]
    lengths = result["target_lengths"]

    assert isinstance(targets, torch.Tensor)
    assert isinstance(lengths, torch.Tensor)
    assert targets.dtype == torch.long
    assert lengths.dtype == torch.long

    assert len(lengths) == 3
    assert lengths.tolist() == [len(t) for t in batch_texts]
    assert targets.numel() == int(lengths.sum())

    # Verify concatenated contents
    expected_ids = []
    for t in batch_texts:
        expected_ids.extend(vocab.encode(t))
    assert targets.tolist() == expected_ids


def test_vocabulary_serialization_json():
    """Verify that TalimVocabulary correctly serializes to and deserializes from JSON."""
    vocab = TalimVocabulary()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        vocab.save(tmp_path)
        loaded = TalimVocabulary.load(tmp_path)

        assert len(loaded) == len(vocab)
        assert loaded.blank_index == vocab.blank_index
        assert loaded.char_to_index == vocab.char_to_index
        assert loaded.index_to_char == vocab.index_to_char
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def test_build_vocabulary_from_labels_with_fallback():
    """Verify build_vocabulary_from_labels falls back cleanly to canonical vocabulary when path is None or missing."""
    vocab_none = build_vocabulary_from_labels(None)
    assert len(vocab_none) == 67

    vocab_missing = build_vocabulary_from_labels(Path("non_existent_directory_12345"))
    assert len(vocab_missing) == 67
