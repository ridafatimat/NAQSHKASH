# Preflight manual review checklist

No human/domain-expert review has occurred. AUTO_VERIFIED is not a substitute.

For each selected matched pair, record reviewer identity, date, image ID and
decision in a new review file; do not overwrite frozen generated annotations.

- Confirm the control has readable Talim and agrees with original structured runs.
- Locate the mutation using row, tier and column; compare the full-resolution zoom.
- Confirm the actual input PNG changed, and contains no caption/answer label.
- Check the wrong-tier glyph is visibly distinguishable from all legal tier aliases.
- Check count-token order/grouping or missing count is genuinely visible.
- Try to describe a legitimate alternate canonical interpretation; if one exists,
  mark EXCLUDED even if the intended label says invalid.
- Flag any reliance on a historical spacing/dialect convention not established by
  the code as NEEDS_DOMAIN_REVIEW, not proven invalid authentic Talim.
- Inspect illumination variants of both valid and invalid images. Marks/order/
  absence must remain interpretable. Lossy/spatial augmentation is not certified.
- Inspect at least one dense/long example and one multi-row example per category.
- Keep preflight as development-design evidence. Never use a sealed protected
  test image for rule design, debugging, selection or calibration.

Approve or revise the scope before running full generation. No expert-reviewed
claim may be added without an actual review record.
