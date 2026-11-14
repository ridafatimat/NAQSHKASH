# E1 handwritten validation baseline â€” PROVISIONAL, not scoreable

The frozen E1 model was run on image-only candidate tier crops from all three original validation photographs. The evaluation reached segmentation, decoding and visual verification. It does **not** yield trustworthy CER or exact-tier accuracy: both existing D1 and the minimal handwritten adaptation fail to isolate complete, correctly assigned tiers. No prediction was scored against an uncertain label.

| Quantity | Result |
|---|---:|
| Original validation photographs attempted | 3 |
| Photographs with E1 candidate inference | 3 |
| Expected logical rows / nonempty reference tiers | 5 / 14 |
| Expected tier slots including one empty lower tier | 15 |
| Candidate tier crops inferred | 15 |
| Verified complete, role-aligned tiers | 0 |
| Sources / tiers scored | 0 / 0 |
| Segmentation or role-alignment failures | 3 sources |
| D1 rejections / visually identified false acceptance | 2 / 1 |
| OCR execution failures | 0 |
| CER / exact-tier accuracy | N/A / N/A |
| Substitutions / deletions / insertions | N/A / N/A / N/A |

N/A is not zero. Fifteen completed forward passes are not fifteen correct recognitions. This is not recognition-only accuracy or full end-to-end accuracy. All three sources and all fifteen candidates are explicitly retained in predictions.csv and metrics.json, with scoring exclusions.

## Fix and integration

Fixed `stage2/preprocessing_improvements/preprocessing/geometry.py` Hough iteration from `lines[:,0]` to `np.asarray(lines).reshape(-1,4)`. OpenCV here returns `(N,4)` rather than the historical `(N,1,4)`. The fix handles both without changing line values or the correction algorithm. No Stage 2E/F/G result files were overwritten. E1 context is imported before D1 to reuse its existing namespace/import compatibility setup; no shared import refactoring was necessary.

Reused frozen D1 config and page preprocessing. D1 rejection alone was not treated as model failure. The isolated adapter detects chromatic blue or local dark ink, rejects obvious border/ruling components, groups large components into candidate count rows and assigns spatial bands above/below them. It receives pixels only: no transcription, JSON, expected row count or token count controls coordinates. Labels are read only after candidate coordinates are fixed. Candidate boundaries remain explicitly unverified. The adapter does not send a whole page as a single tier and is not an approved production segmenter.

## Per-source findings

- **all_003191**: 6 expected nonempty tiers, 6 candidates inferred, 0 scored. D1 falsely accepts a layout whose bands/roles do not match the visible writing. Adaptation separates two count rows but lower row 2 collapses to a narrow pair of strokes; other small marks overlap the count band. Full tier contents are not isolated.
- **all_003556**: 3 expected nonempty tiers, 3 candidates inferred, 0 scored. D1 rejects unresolved vertical merge. Adapted count candidate clips the first glyph and omits the right-hand glyphs; curved count descenders enter the lower candidate. Faint blue ink is not consistently recovered.
- **all_009197**: 5 expected nonempty tiers, 6 candidates inferred, 0 scored. D1 rejects no legal layout. Adaptation splits count descenders and sloped adjacent tiers across boundaries; it creates an extra lower candidate for row 2, whose reference lower tier is empty. A raw pen fragment is not proof of an annotation error.

## Predictions and error evidence

Fourteen candidate predictions are `Y`; one is `X`. This concentration is observed output, not a valid confusion matrix or proof that particular handwriting symbols were mistaken for Y/X. Count-role candidates also decode as Y, indicating poor raw outputs on these candidate inputs. Their incorrect role/boundary alignment prevents legitimate count-error accounting. Do not substitute the candidate_expected column for a verified reference correspondence.

Segmentation is a demonstrated contributor: clipping, omitted faint notation, extra pen-fragment candidates, notebook slope and overlap between count descenders and upper/lower notation are visible in segmentation_previews. Handwriting domain shift may also contribute to the collapsed predictions, but this run cannot separate it from segmentation. Wrong copied source labels have not been established. File-level TXT/JSON agreement is not physical handwriting verification.

error_examples contains the 15 final grayscale crops. predictions.csv pairs them with the actual prediction and tentative expected label. Because none has established semantic alignment, they are unscored diagnostic examples, not verified character-error examples. The coordinates JSON contains original-image candidate boxes, processed-image D1 boxes, all D1 diagnostics and per-page outcomes. D1 previews use processed coordinates; candidate previews use original coordinates.

## Model, labels and metric definitions

Checkpoint: `stage2/stage2e_finetuning/checkpoints/selected_model/best.pt`.
SHA-256 verified before and after inference, and in tests: `27f2da50795e2fca8f327afefdce10645f3e60e5320060a8cee93994fabdcdf1`.
Existing Stage 2E context loads CRNN(66), strict weights, tokenizer/character mapping and vocabulary hash. Existing tensor normalization is grayscale float32/255; existing aspect-preserving bilinear resize sets height 64. Existing greedy CTC decoder is reused. Direct one-tier inference has no batch padding. No weights or tokenizer were changed.

Reference upper/lower and count_line strings map to upper/lower/count roles. Stage 2E removes leading layout offsets using lstrip(' ') while retaining internal spaces; candidate_expected follows that convention. Empty lower tiers are not normal nonempty training targets. Once correspondence is verified, Stage 2E CER is total substitutions+deletions+insertions divided by reference characters, and exact accuracy is complete-string equality over scored tiers. Those denominators cannot legitimately be populated from misaligned crops, so metric values remain null.

## Regression validation and scope

Five focused tests passed: both Hough shapes (including reproducing the historical exception), no-line fallback, historical clean behavior, image-only candidate segmentation/blank input, and scoring/checkpoint safeguards. The historical test compares full preprocessing pixels, matrices, roles, rejection and diagnostics against the pre-fix function using the old supported Hough shape, on two clean synthetic validation references plus one existing historical clean positive-development tier. Outputs are identical. This is representative regression coverage, not a rerun of Stage 2F.

Only all_003191, all_003556 and all_009197 handwritten originals were processed. No augmented images or protected test images were inspected or inferred. No training, fine-tuning, model changes, Git/GitHub operations or protected Stage 2F evaluation. Prior result directories are preserved. The sole shared-source change is the geometry compatibility fix.

## Readiness and exact next steps

E1 is not ready for valid handwritten fine-tuning from this page-level dataset yet. Its frozen forward pass works; tier isolation and source-label correspondence remain unresolved. Do not use candidate crops as automatically approved training pairs.

1. Have a Talim reader verify the physical writing against the references for these three validation sources, recording discrepancies without changing the originals.
2. Annotate complete upper/count/lower regions in the original photographs independently of OCR output. For overlapping strokes, use region masks or separate verified tier transcription; simple horizontal cuts are insufficient. Resolve missing/faint/clipped notation explicitly.
3. Reuse E1 and the existing Stage 2E metrics only on those verified tier pairs. Report coverage and page failures alongside conditional recognition metrics, and mark results PROVISIONAL until semantic verification is complete.
4. Use validation data only to validate a generalized handwritten segmenter. Keep protected test sources untouched. Prepare verified training tier pairs before considering a separately authorized fine-tuning run.
