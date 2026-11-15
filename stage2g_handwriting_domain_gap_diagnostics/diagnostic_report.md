# Stage 2G handwriting domain-gap diagnosis — PROVISIONAL

Frozen E1 only. No training, model-weight changes, label edits, protected-test inference or Git operations. All 14 final validation tiers from three sources are fixed. Independent Talim-reader semantic verification remains incomplete.

## Matched preprocessing controls

| Control | Tiers | CER | CER without whitespace | Exact |
|---|---:|---:|---:|---:|
| synthetic_standard | 60 | 0.0000% | 0.0000% | 60/60 |
| synthetic_paired_D1 | 14 | 0.0000% | 0.0000% | 14/14 |
| synthetic_handwriting_convention | 14 | 3.0928% | 2.8169% | 11/14 |

The paired synthetic references have exactly the same source/row/role targets as the 14 handwriting tiers. D1 is the frozen Stage2E page preprocessing; renderer-slot masks independently check retention and contamination. Segmentation is assessed before scoring. The handwriting-convention control uses known synthetic masks and the documented row-union/padding/keep-faint conventions with H=count-ink height and zero deskew. It isolates crop/render geometry and is not an end-to-end replay of unavailable photograph masks or segmentation heuristics. No synthetic reference from protected sources is opened. The 60 additional established Stage2E clean crops are deterministic 20-per-role controls, not a random population estimate.

## Frozen-E1 handwritten ablations

| Transformation | CER | No-whitespace CER | Exact tiers | Mean blank argmax fraction | Mean CTC NLL/target char |
|---|---:|---:|---:|---:|---:|
| A_current | 74.2268% | 88.7324% | 0/14 | 0.939 | 8.106 |
| B_contrast | 73.1959% | 88.7324% | 0/14 | 0.928 | 7.741 |
| C_otsu | 75.2577% | 90.1408% | 0/14 | 0.933 | 8.050 |
| D_dilation | 69.0722% | 80.2817% | 0/14 | 0.941 | 6.793 |
| E_tight_vertical | 75.2577% | 87.3239% | 0/14 | 0.945 | 7.886 |
| F_glyph_height | 75.2577% | 87.3239% | 0/14 | 0.948 | 7.580 |

Each B–F starts independently from A. No transforms are stacked. Fixed frozen weights, vocabulary, normalization, CTC lengths and greedy decoder. Transformation recipes and synthetic-only height targets are in configuration.json. Otsu can remove faint ink; dilation can merge dots; vertical remapping changes glyph aspect. These are diagnostic ablations, not automatically safe production changes.

## Rolewise current-input errors

| Role | Tiers | CER | No-whitespace CER | S / D / I | Exact |
|---|---:|---:|---:|---:|---:|
| upper | 5 | 61.2903% | 76.4706% | 10 / 7 / 2 | 0/5 |
| count | 5 | 92.6829% | 92.6829% | 26 / 11 / 1 | 0/5 |
| lower | 4 | 60.0000% | 92.3077% | 9 / 6 / 0 | 0/4 |

All transformations and synthetic controls have rolewise rows in rolewise_error_analysis.csv. Minimum-edit ties use the Stage2E substitution/deletion/insertion preference. Confusion counts are alignment diagnostics, not a Talim semantic judgment.

## CTC evidence

- handwriting A: time steps 73–273; required steps 1–17; minimum T/target 10.40; mean blank posterior 0.938; mean blank argmax fraction 0.939; empty predictions 0/14.
- matched synthetic D1: time steps 81–364; required steps 1–17; minimum T/target 16.00; mean blank posterior 0.931; mean blank argmax fraction 0.931; empty predictions 0/14.

All scored samples have enough true time steps, including repeated-target blank separation. Insufficient CTC length is ruled out for these inputs. Blank dominance alone is expected in CTC and cannot establish deletion causality. Full per-frame 66-class probabilities and argmax paths are saved in posteriors/, alongside entropy, target-class mass, target-sequence negative log likelihood, repeated nonblank frames, run counts and decoded repeats in predictions_and_ctc_behavior.csv. A width-10 prefix beam without a language model gives 73.1959% CER / 87.3239% no-whitespace CER / 0/14 exact. This separate untuned decoder check does not change any primary ablation. Beam width is finite and cannot rule out every decoder improvement.

## Distribution measurement definitions

input_distribution_comparison.csv contains per-image geometry, padding, nonspace-target density, grayscale intensity, faint-ink fraction, contrast, stroke-width proxy, E1 width and nominal CTC feasibility; summary CSV gives per-role medians. Ink is <250; component area >=2 pixels; stroke proxy is twice distance-transform radius at local medial ridges. These proxies depend on threshold and do not equate multipart connected components to semantic glyphs. Blank background contrast is measured on already rendered crops, not original paper photos. Faint values 250–254 are excluded from metric masks but preserved in geometry transforms. E uses the complete <255 vertical span so no retained nonwhite pixel is deliberately cropped.

## Reproduction and integrity

Run `python -B diagnose.py` using the recorded runtime and the existing project paths. Writes are confined to this diagnostics directory. A completed diagnostic_report.md causes refusal to overwrite: for a fresh repeat, copy script to a new isolated output folder and change OUT to that folder. This script reads the exact Downloads release hash and uses its already verified imported final dataset. It never loads earlier handwritten ZIPs. Python dependencies are the existing CPU PyTorch/OpenCV/NumPy/Pillow environment. No install or network access is needed.

integrity_checks.json records weight-state equality, no parameter gradients, exact reproduction of the previous 14 baseline predictions, sufficient CTC lengths, fixed labels/samples, and before/after hashes for every previous experiment file plus selected E1 and directly used labels/images/software. Source hash and runtime/config records are included. Earlier Stage2E/F files are used read-only; Stage2F images are not loaded.

## Diagnostic conclusions and next-experiment decision

**Primary diagnosis: an appearance-domain mismatch (stroke shape, thickness and faintness), with a secondary preprocessing contribution; the tested geometry conventions and decoder do not explain most of the failure.** This is an inference from matched controls, not a causal partition of error percentages. Exact-label synthetic references remain at 0% CER under D1 and 3.09% under the documented handwriting-convention emulator, while actual handwriting is 74.23%. Synthetic glyph identity is therefore recognized on these targets; crop/render convention alone produces a much smaller failure. Contrast alone and vertical normalization do not close the gap. Unverified writer copying/target semantics remain a confounder that image-label file agreement cannot resolve. No original handwriting page segmenter is evaluated, so this does not establish end-to-end page segmentation quality.

### Measured appearance and geometry differences

| Role | Synthetic / handwriting median stroke-width proxy at E1 scale | Synthetic / handwriting ink median | Synthetic / handwriting component-height proxy at E1 scale | Synthetic / handwriting resized width |
|---|---:|---:|---:|---:|
| upper | 10.97 / 4.44 | 3 / 183 | 20.11 / 10.32 | 644.0 / 672.0 |
| count | 7.68 / 2.41 | 3 / 181 | 40.35 / 19.20 | 451.0 / 403.0 |
| lower | 11.30 / 4.09 | 3 / 169 | 17.37 / 10.24 | 743.5 / 581.5 |

Stroke widths are geometric proxies, not calibrated pen widths; component heights are not glyph heights because multi-dot/line notation can have several components. Overall vertical occupied fractions and E1 widths are broadly similar on these matched samples, while darkness, local thickness and component morphology differ substantially. Handwriting has larger native pixels but E1 normalizes those away. Its foreground is much lighter; already-rendered backgrounds are essentially white in both domains. Full dimensions, aspect ratios, density and padding are in the distribution CSVs. Shape variants and the persistence of errors after contrast/height changes support an appearance gap; this experiment cannot fully separate shape from grayscale effects or label correctness.

### Which preprocessing helps?

Dilation is the only tested change with a notable aggregate gain: 74.23% to 69.07% CER (72 to 67 edits; five fewer edits), and 88.73% to 80.28% no-whitespace CER. It improves 5 tiers, worsens 2, and leaves 7 unchanged. Still 0/14 exact; this is exploratory and not evidence of generalization. Its mean blank fraction rises slightly rather than falls, so the improvement cannot be attributed simply to reducing blanks. Contrast yields only one fewer edit and no no-whitespace gain. Otsu, vertical tightening and synthetic-role ink-height normalization yield one more space-sensitive edit than baseline. No combined transforms were tested.

The resize/uint8-handoff identity control reproduces all 14 baseline predictions; thus the observed dilation gain is not explained by its uint8 handoff on these samples. Applying the same dilation to 60 previously perfect synthetic controls yields 1.4315% CER and 54/60 exact. Dilation must therefore be calibrated on training data and preserve dots/gaps; it is not automatically a universal inference default.

### Most affected roles and character behavior

Count is worst by primary CER: 92.68% versus upper 61.29% and lower 60.00%. Without whitespace, lower is almost equally poor (92.31%; upper 76.47%), so the lower headline benefits from easier alignment spaces. Repeated count confusion is concentrated on `i`: target a→i five times, d→i four, b→i three and c→i twice under the deterministic alignment. These are OCR-class confusions, not claims about the Talim meaning of handwriting. Whitespace accounts for 26/97 targets; removing it does not remove the domain gap.

Across the five count outputs, 20/31 decoded characters are i, against 3 target i characters. The repeated decoded labels inspected are all separated by blank runs: True. ctc_nonblank_runs.csv records their exact positions and posterior confidence. This is repeated character emission by the model, not failure to collapse adjacent identical CTC frames. Original mean nonblank-frame confidence for count is 0.782; high confidence can still be wrong out of domain.

The model assigns very low probability to the correct handwriting sequence (mean per-tier target CTC NLL/character 8.106, versus the much smaller matched-synthetic control value in control_results.json). All original targets have enough time steps; no outputs are empty. A blank-heavy posterior occurs on correctly recognized synthetic tiers too. The separate width-10 beam check recovers only one edit, preserving 24 aligned deletions, so greedy search is not the main demonstrated bottleneck. This does not prove a cause for each deletion: target probability mass may be spread across wrong identities, positions or blank, and multipart handwriting and copying errors remain possible.

### Is a second fine-tuning run justified now?

**Not yet as another unconstrained fine-tuning attempt.** The evidence is sufficient to design an appearance-focused pilot, but not to certify labels, select a universal preprocessing change, or claim generalization from three validation sources. Before executing it: obtain independent Talim-reader checks of the handwritten count copying and notation targets (prioritize all five validation counts and the 57 training counts, then all upper/lower targets); verify the 165 training-only crops for dilation-induced merging at model scale; calibrate any stroke policy using training images only. Keep the 14 validation labels and sources fixed and the protected test closed. This diagnosis does not create a new training set or authorize a test evaluation.

### Exact next strategy to test after those checks (proposal only)

One predeclared appearance-adaptation pilot, initialized anew from frozen E1; do not continue epoch 2 of the rejected model. Use only the verified 165 handwritten training tiers, distinguishable synthetic replay, no validation training, seed 42 and a fixed two-epoch ceiling. Preserve original ocr_text/internal whitespace. Use effective batch four (two handwritten plus two replay, gradient accumulation, frozen BatchNorm statistics). Role-stratify handwritten ordering without oversampling count until the first pass covers all 165 tiers.

Use AdamW with weight decay 1e-4 and gradient clipping 1.0; freeze the BiLSTM and classifier for this first appearance-focused probe, and train the CNN at 1e-5. This is a hypothesis derived from the appearance mismatch, not a validated optimal schedule. Use a 50/50 identity versus one-output-pixel cross dilation augmentation for handwritten TRAINING images only if visual QA confirms dots/gaps survive; keep validation inference at unchanged A for the primary comparison and report fixed D as a separate diagnostic. Do not adopt Otsu, vertical warps or contrast+dilation stacks. Replay uses original synthetic tiers; require original synthetic regression cohorts after every epoch. Do not generate new labels or pretend the current threshold proxy yields verified glyph segmentation.

Include a frozen-E1 distillation anchor on replay: add 0.1 times the mean-over-true-time-step KL divergence KL(teacher probabilities || student probabilities) to CTC loss, temperature 1.0. The unmodified frozen E1 is the teacher; no teacher updates. Keep exact time widths and blank index unchanged. This is a proposed forgetting-control mechanism, not a result of this diagnosis. Checkpoint selection should retain E1 unless handwritten CER improves while every clean cohort increases by <=0.001 absolute, clean exact drops <=0.01, positive-edge CER increases <=0.005 absolute, and long-positive CER increases <=0.005. Report every epoch, role, no-whitespace score and sample exposure; reject any candidate violating a gate. Do not expand epochs or search hyperparameters on the 14 tiers.

No second training run was started. Frozen E1 remains selected.
