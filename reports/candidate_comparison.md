# Stage 2D candidate comparison

D1 is frozen from original tuning parents only. Validation below was already exposed in Stage 2C and is a development regression comparison, not an independent test. C4 and C5 are unchanged; their segmentation was replayed for all pages and OCR was recomputed from saved predictions. D1 uses the original frozen model and decoder. Exact-pixel-hash OCR reuse cannot select a segmentation.

| Scope | Candidate | Correct/pages | Wrong | Rejected valid | Correct coverage | Conditional CER | Exact tiers | Common pages | Common CER |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| normal clean_regression | C4 | 96/100 | 4 | 0 | 96.000% | 0.177% | 99.046% | 96 | 0.177% |
| normal clean_regression | C5 | 96/100 | 3 | 1 | 96.000% | 0.177% | 99.046% | 96 | 0.177% |
| normal clean_regression | D1 | 98/100 | 0 | 2 | 98.000% | 0.173% | 99.070% | 96 | 0.177% |
| general validation | C4 | 135/150 | 15 | 0 | 90.000% | 1.919% | 92.699% | 134 | 1.911% |
| general validation | C5 | 134/150 | 13 | 3 | 89.333% | 1.855% | 93.096% | 134 | 1.855% |
| general validation | D1 | 149/150 | 0 | 1 | 99.333% | 2.546% | 90.819% | 134 | 1.855% |
| positive validation | C4 | 94/138 | 41 | 3 | 68.116% | 27.532% | 90.708% | 92 | 13.076% |
| positive validation | C5 | 94/138 | 13 | 31 | 68.116% | 27.522% | 90.708% | 92 | 13.076% |
| positive validation | D1 | 137/138 | 0 | 1 | 99.275% | 12.371% | 92.355% | 92 | 13.076% |

## Frozen selection and reliability
[
  {
    "candidate": "E1.3_H0.85_M0.2",
    "correct": 830,
    "wrong": 3,
    "positive_correct": 535,
    "general_correct": 295,
    "clean_correct": 278,
    "safety_gate": true,
    "wrong_reference": 75,
    "correct_reference": {
      "positive": 327,
      "general": 271
    }
  },
  {
    "candidate": "E1.3_H0.85_M0.4",
    "correct": 804,
    "wrong": 3,
    "positive_correct": 510,
    "general_correct": 294,
    "clean_correct": 268,
    "safety_gate": true,
    "wrong_reference": 75,
    "correct_reference": {
      "positive": 327,
      "general": 271
    }
  },
  {
    "candidate": "E1.3_H0_M0.2",
    "correct": 782,
    "wrong": 22,
    "positive_correct": 482,
    "general_correct": 300,
    "clean_correct": 262,
    "safety_gate": true,
    "wrong_reference": 75,
    "correct_reference": {
      "positive": 327,
      "general": 271
    }
  },
  {
    "candidate": "E1.5_H0.85_M0.4",
    "correct": 767,
    "wrong": 40,
    "positive_correct": 513,
    "general_correct": 254,
    "clean_correct": 257,
    "safety_gate": false,
    "wrong_reference": 75,
    "correct_reference": {
      "positive": 327,
      "general": 271
    }
  },
  {
    "candidate": "E1.5_H0.85_M0.2",
    "correct": 792,
    "wrong": 41,
    "positive_correct": 538,
    "general_correct": 254,
    "clean_correct": 266,
    "safety_gate": false,
    "wrong_reference": 75,
    "correct_reference": {
      "positive": 327,
      "general": 271
    }
  },
  {
    "candidate": "E1.5_H0_M0.2",
    "correct": 742,
    "wrong": 62,
    "positive_correct": 485,
    "general_correct": 257,
    "clean_correct": 250,
    "safety_gate": false,
    "wrong_reference": 75,
    "correct_reference": {
      "positive": 327,
      "general": 271
    }
  }
]

The safety gate requires at least 95% of C5 tuning correct scanned pages separately for positive/general, while halving C5 tuning scanned wrong acceptances. No OCR score or validation outcome chooses the candidate. Historical engineering targets are not tuning permissions. Six configurations and one documented training-parent gap-rule correction were tested; first-attempt outputs and its focused failure remain in reports/attempt1.

All variants are correlated by source parent. See parent_results.csv, d1_change_counts.csv and candidate_summary.csv for per-parent coverage, wrong acceptance and rejection counts, including overlapping parent sets. Common-subset OCR uses only complete correctly segmented, CTC-feasible pages shared by C4/C5/D1 within each cohort. It differs from the historical six-candidate intersection, so its denominator is reported explicitly. Conditional OCR excludes wrong/rejected pages; page_exact_all includes them as unsuccessful.

Accepted row-count accuracy and exact ordered tier-role structure are separate machine-readable columns. Rejected cases are never credited as successful segmentations. hypothesis_row_count_accuracy_all separately evaluates the raw best row-count hypothesis even if confidence later abstains; missing hypotheses count as wrong. Component preservation is measured independently on ideal source components >=3px; per-page geometric loss and approximate processed-ink support are in evaluation_v1/pages.csv. This is not proof that all real scanned dot shapes are visually correct.

## Interpretation
D1 targets grouping and role confidence; photometric normalization, denoising, geometry, model and decoder are inherited unchanged. Remaining wrong acceptances must stay visible and prevent automatic deployment approval. Positive conditional CER can remain dominated by the long exposed parent; no target spaces are removed and no failure is omitted. Do not interpret synthetic scanned-style development results as handwriting generalization.

See failure_analysis.md for root-cause evidence, fine_tuning_readiness.md for verified training eligibility, and galleries/index.html for original/processed/full-resolution examples.

## Important regressions and subset effects

D1's scanned tuning result is 830/846 correct, with three wrong acceptances, all from edge_00310. C5 was 598/846 correct with 75 wrong acceptances. The frozen safety rule passed before validation.

On the already exposed scanned validation comparison, D1 is 137/138 positive and 149/150 general correct, with zero wrong acceptances and one rejection in each dataset. Every one of the 46 positive and 50 general parents has at least one correctly segmented variant; 45/46 positive and 49/50 general parents have all three scanned variants correct. These are correlated development variants, not 288 independent source parents.

Do not treat zero scanned validation wrong acceptances as universal safety: across all 1,612 pages, D1 has **1,585 correct acceptances, five wrong acceptances and 22 rejections**. Four wrong pages belong to tuning parent edge_00310 (one clean and all three scans); one wrong page is exposed-validation clean edge_00307. These are valid lower-symbol layouts. Visual review shows a disconnected lower symbol split into multiple count rows. No post-validation fix or reselection was made.

Compared with C5, D1 newly rejects five previously correct general tuning scans from two parents (all_005744, all_009950), plus the clean all_009950 page. It makes the clean and two scanned edge_00310 pages worse; the third scan was already wrong under C5. These losses remain in d1_change_counts.csv/regression_analysis.json.

The one lost positive scanned validation page is **edge_00323__aug_02_perspective**, rejected for unassigned foreground. C4/C5 scored 479 errors on its 758 characters. Therefore the reduction from ~27.53% to **12.371% positive conditional CER** is substantially a difficult-page rejection/subset effect, not a recognition improvement. On the shared C4/C5/D1 subset of 92 positive pages (1,866 characters), all three have **13.076% CER**. The retained long clean page still has 112 missing internal spaces.

Expanded general conditional CER increases from **1.855% C5 to 2.546% D1**. D1 retains every one of C5's 134 correctly segmented general scans, adding 15 harder pages. The 134-page common subset is unchanged at **1.855% CER**; the additional 671 characters contribute 54 errors. Recognition on the shared subset did not worsen, but newly covered distorted images remain difficult for the frozen CRNN.

Normal clean regression improves from 96 to 98 correct pages, with wrong acceptances reduced from three to zero relative to C5. No previously correct normal clean page is lost. Two previously wrong pages are corrected; another wrong page is rejected. Common-subset clean CER is unchanged at 0.177% on 96 shared pages. One additional general tuning clean page is rejected, and positive tuning edge_00310 regresses; clean regression is therefore not a universal no-regression claim.

## Continuity with Stage 2C's historical common subset

The historical six-candidate intersection is retained completely by D1: 32 positive scanned validation pages (96 characters), 111 general scanned pages (4,305 characters), and 88 normal clean pages (3,667 characters). On these identical pages: positive CER remains 3.125% for C0/C4/C5/D1; general CER is 6.411% C0, 1.835% C4 and 1.765% C5/D1; clean CER remains 0.191%. See historical_common_subset.csv. These denominators differ from the broader three-candidate common subset above.

## Component and structural verification

The identical 90-page diagnostic has 70 structurally aligned pages for C4, 70 for C5 and 87 for D1. No geometric component losses were found among aligned cases. The approximate processed-ink content diagnostic finds four weak components for each historical candidate and zero for D1; this is not a manual pixel-shape guarantee.

An additional intended-tier audit over all 1,585 structurally aligned D1 pages checks **65,881 components >=3px**, finding zero components below 95% retention in their intended crop. This is stronger than a union-of-crops test: a dot in the wrong tier cannot be certified merely because some crop contains it. A dedicated adversarial test demonstrates that aggregate 98% ink rules alone can miss that error.

Raw row-count hypothesis accuracy, accepted row coverage/accuracy, exact tier-role structure, proposal coverage, wrong acceptances, valid rejections, page-exact/all, role-specific OCR, actual augmentation recipes and per-parent results are saved separately. Every denominator is explicit. Component masks derive from the frozen synthetic renderer and recorded transformations and approximate blur/interpolation; reference components <3px are not certified.

## Fine-tuning recommendation

The verified dataset subset contains **714 images from 179 training parents and 1,794 tier crops** (1,655 unique crop hashes; 145 distinct target strings). Fourteen images from five training parents need manual review; these are not eligible. The 184 exposed internal-validation images from 46 parents remain excluded from training. Ready and review parent counts overlap because variants can have different outcomes. All 228 positive source-parent transcriptions were independently rechecked against TXT, JSON and canonical Talim rules. Human-reviewed count is zero.

Proceed next to a controlled loader/resource validation and a separately approved fine-tuning pilot using only the verified subset, with parent isolation, a clean-training replay strategy, a frozen validation/stopping protocol and explicit spacing/long-sequence metrics. The current data are small targeted edge cases, not proof that fine-tuning will help. Keep the 14 review pages excluded until resolved; do not automatically train them from D1 acceptance.

D1 is **not deployment-approved**: residual valid-layout mistakes, exposed validation, synthetic-only degradation and long OCR spacing/glyph errors remain. Obtain prospective independent evaluation before deployment/generalization claims. No model training, checkpoint changes, original dataset changes, Git operations, negative-example training or handwriting training occurred.
