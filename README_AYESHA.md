# NAQSHKASH Preprocessing & Token Vocabulary — Ayesha Amer

**Author:** Ayesha Amer  
**Team:** NAQSHKASH FYP  
**Phase:** Preprocessing & Data Representation (Days 1–4 of 14-Day Plan)

---

## 1. Overview & Scope

This package implements the foundational preprocessing stages and token representation for Kashmiri Carpet Talim document recognition:
- **Day 1 Scope:** Content margin detection and bounding-box cropping (`preprocessing.crop`).
- **Day 2 Scope:** 3-tier logical row detection and row separation (`preprocessing.rows`).
- **Day 3 Scope:** Grayscale conversion, 64px aspect-preserving resizing (`cv2.INTER_AREA`), float32 normalization in `[0.0, 1.0]`, and model preparation pipeline (`preprocessing.normalize`, `preprocessing.model_prep`).
- **Day 4 Scope:** Token vocabulary and PyTorch CTC label encoder (`data.vocabulary`, `data.encoder`, `data/vocab.json`).

### Strict Scope Boundaries
- **In Scope (Days 1–4):** Standalone margin cropping, 3-tier row grouping, 64px normalization, complete 67-class Talim vocabulary (30 symbols + 34 count glyphs + space + newline + CTC blank), and PyTorch `nn.CTCLoss` label encoding.
- **Teammate Scopes:** 
  - **Rida (Days 4–5):** `TalimDataset`, `DataLoader`, batch collation, and variable-width padding (`data.dataset`, `data.dataloader`, `data.collate`).
  - **Hareem (Days 4–6):** CRNN architecture, BiLSTM layers, CTC Loss training loop (`models/`).
  - **Ayesha (Days 5–6):** CTC decoding support and greedy CTC decoder (`data.ctc_decode`).

---

## 2. Kashmiri Carpet Talim Row Definition & Flattening Convention

In Talim notation, **one logical row is defined as a 3-tier block**:
1. **Upper Tier (optional):** UP-direction color symbols (e.g. `CN`, `A`, `Q`).
2. **Center Tier (mandatory):** Count glyphs indicating stitch quantities (e.g. `jjj`, `eijf`, `l`).
3. **Lower Tier (optional):** DOWN-direction color symbols (e.g. `R`, `_`, `]`, `T`).

```text
+-------------------------------------------------------------+
| Upper Tier:  [ C N ]              [ Q E ]       (Symbols)   |
| Center Tier: [ j j j ]            [ j e i f ]   (Counts)    | ===> 1 Logical Row
| Lower Tier:  [       _ ]          [     R _ ]   (Symbols)   |
+-------------------------------------------------------------+
  <---------------- Inter-row Whitespace Gap ---------------->
+-------------------------------------------------------------+
| Upper Tier:  [   N ]                                        |
| Center Tier: [ j j j ]                                      | ===> Next Logical Row
| Lower Tier:  [ ]   U ]                                      |
+-------------------------------------------------------------+
```

### 3-Line Label Flattening Convention
Label text files flatten the multi-tier notation top-to-bottom:
$$\text{Upper Tier Line} \xrightarrow{\backslash\text{n}} \text{Center Tier Line} \xrightarrow{\backslash\text{n}} \text{Lower Tier Line} \xrightarrow{\backslash\text{n}\backslash\text{n}} \text{Next Logical Row}$$

---

## 3. Token Vocabulary Specification (Day 4)

Total Classes: **67** (including CTC blank token at index 0):
- **CTC Blank:** Index `0` (`<blank>`). 0 is never assigned to any valid character.
- **30 Trusted Symbols:**
  - 16 UP Symbols: `['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'P', 'Q']`
  - 14 DOWN Symbols: `['R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '_']`
- **34 Authentic Count Glyphs (1–259):**
  - Ones (1–9): `'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'`
  - Tens (10–190): `'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|'`
  - Hundreds (200–250): `"'", '(', ')', '*', '+', ','`
- **2 Formatting Characters:** `' '` (space), `'\n'` (newline).

---

## 4. Package Architecture & Public APIs

```text
preprocessing/
├── __init__.py        # Public API exports
├── deskew.py          # Rida: Rotational deskewing and skew estimation
├── crop.py            # Margin removal and content bounding-box extraction
├── rows.py            # 3-tier logical row detection and extraction
├── pipeline.py        # Shared: Base crop-and-separate pipeline
├── normalize.py       # Grayscale conversion, 64px resizing, and [0, 1] normalization
└── model_prep.py      # End-to-end model-ready row preprocessing pipeline

data/
├── vocabulary.py      # TalimVocabulary, build_vocabulary_from_labels, normalize_label_text
├── encoder.py         # CTCLabelEncoder (text -> torch.LongTensor, batch encode)
└── vocab.json         # Canonical 67-class token mapping JSON
```

---

## 5. Usage Example (Day 4 Vocabulary & Encoder)

```python
from data.vocabulary import TalimVocabulary, build_vocabulary_from_labels
from data.encoder import CTCLabelEncoder

# 1. Initialize Canonical Vocabulary (or build from labels)
vocab = TalimVocabulary()
print(f"Total Classes (incl. blank): {len(vocab)}")  # 67
print(f"CTC Blank Index: {vocab.blank_index}")       # 0

# 2. Encode & Decode Text
text = "CN\njjj\n  _\n\n"
encoded_ids = vocab.encode(text)
decoded_text = vocab.decode(encoded_ids)
assert decoded_text == text

# 3. Batch Encode for PyTorch CTCLoss
encoder = CTCLabelEncoder(vocab)
batch = ["CN\njjj\n  _", " N\njjj\n] U"]
batch_encoded = encoder.encode_batch(batch)

targets = batch_encoded["targets"]               # 1D LongTensor of shape (sum(lengths),)
target_lengths = batch_encoded["target_lengths"] # 1D LongTensor of shape (B,)
```

---

## 6. Verification & Tests

### 1. Pytest Suite
```bash
# Run Ayesha Day 4 Vocabulary & Encoder tests:
python -m pytest tests/test_ayesha_vocab_encoder.py -v

# Run all Preprocessing tests (Days 1-3):
python -m pytest tests/test_preprocessing_crop.py tests/test_preprocessing_rows.py tests/test_preprocessing_pipeline.py tests/test_preprocessing_normalize.py tests/test_preprocessing_model_prep.py -v
```

### 2. Dataset Verification
- Verified on 1,098 train/validation label files (57,705 characters):
  - **Unknown tokens:** 0 (0.00%)
  - **Round-trip accuracy:** 100.00%
  - **All 66 non-blank classes active in dataset.**
