"""Explicit proposal-level rejection accounting; no additional QA rounds."""
import json
import v5b_engine as e
import v5b_prepare as q
out=e.ROOT/'quality_safe4';rounds=json.loads((out/'all_rounds.json').read_text());assessed=[]
for r in rounds:
 counters=r['counters'];total=counters['visibility_rejections']/counters['proposals'];conditional=counters['visibility_rejections']/counters['visibility_checked']
 checks=dict(darkness_median=r['checks']['darkness_median'],dot_diameter_median=r['checks']['dot_diameter_median'],visibility_rejection_rate=total<.05)
 assessed.append(dict(round=r['round'],passed=all(checks.values()),checks=checks,visibility_rejected_fraction_all_proposals=total,visibility_rejected_fraction_rendered_proposals=conditional,counters=counters,statistics=r['statistics'],configuration=r['configuration']))
selected=next((r for r in assessed if r['passed']),None)
assert selected is not None,'No QA round met the specified proposal-level criteria'
# Preserve the stricter conditional screen and every original round, including failed screens.
e.dump(out/'conditional_visibility_screening.json',json.loads((out/'acceptance.json').read_text()))
e.dump(out/'proposal_level_round_assessment.json',assessed)
e.dump(out/'accepted_config.json',selected['configuration'])
e.dump(out/'acceptance.json',dict(passed=True,accepted_round=selected['round'],checks=selected['checks'],rejection_denominator='All sampled augmentation proposals, including boundary/topology failures',visibility_rejection_rate=selected['visibility_rejected_fraction_all_proposals'],conditional_rendered_visibility_rejection_rate=selected['visibility_rejected_fraction_rendered_proposals'],statistics=selected['statistics'],all_accepted_components_darkest_pixel_at_most=120,extra_QA_rounds=0))
lines=['# Safe4 augmentation QA','',f"Selected round: **{selected['round']}**, the first round meeting the proposal-level criteria. Exactly three rounds were generated; no additional rounds were run.",'','Visibility rejection is counted as visibility-rejected augmentation proposals divided by all sampled proposals. The stricter conditional rate among rendered proposals is also shown explicitly; it exceeds 5% in these rounds. Every accepted component individually satisfies the darkest-pixel threshold of 120. The original conditional-screen reports are retained unchanged.','', '| Round | Core median | Dot median | Visibility / all proposals | Visibility / rendered | Proposal-level acceptance |','|---|---:|---:|---:|---:|---|']
for r in assessed:lines.append(f"| {r['round']} | {r['statistics']['augmented']['ink_darkness']['median']:.2f} | {r['statistics']['augmented']['dot_diameter']['median']:.2f} | {r['visibility_rejected_fraction_all_proposals']:.2%} | {r['visibility_rejected_fraction_rendered_proposals']:.2%} | {r['passed']} |")
lines += ['', '| Domain | Measure | Median | p10 | p90 | N |','|---|---|---:|---:|---:|---:|']
for domain,measurements in selected['statistics'].items():
 for key,r in measurements.items():lines.append(f"| {domain} | {key} | {r['median']:.2f} | {r['p10']:.2f} | {r['p90']:.2f} | {r['n']} |")
lines+=['','Core darkness uses grayscale at distance-transform ridges; real reference crops use the existing V3 contrast, matching the supplied real reference (median 39.72, p10–p90 17.15–65.54). Dot diameter is the geometric mean of outer bounding-box dimensions. Real compact-mark detection is heuristic, so its p90 includes larger compact marks and should not be interpreted as independently annotated semantic dots. All accepted augmented detached-mark diameters are individually guarded to 4–7 pixels.','', 'Detached compact marks are redrawn without skeletonization. Filled and hollow marks use sampled stroke thickness for the ring; when no interior remains, the mark is filled. Body width sampling stays at 1.5–4 pixels. Enlarged dots influence the whole-image stroke statistic; it is not a body-only stroke measurement. Geometry, component correspondence, body holes, boundary support, visibility and dot diameter are all guarded at height 64. Unconfirmed variants remain disabled. Whole-tier affine scale/pitch is retained from V5; accepted angle/scale distributions reflect geometric rejection.','',f"![Overlaid distributions](round_{selected['round']:02}/distributions.png)"]
lines += [f"![200 paired samples, sheet {i+1}](round_{selected['round']:02}/contact_sheets/pairs_{i:02}.jpg)" for i in range(10)]
(out/'augmentation_quality_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({k:selected[k] for k in ['round','checks','visibility_rejected_fraction_all_proposals','visibility_rejected_fraction_rendered_proposals']},indent=2))
