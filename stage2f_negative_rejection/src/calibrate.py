"""Manual threshold selection from frozen development outputs only."""
import argparse
import math
from common import *
from evaluate_rejection import load_baseline
from rejection_pipeline import decide


def calibrate(items, lineage, max_review_fraction=0.05, allow_positive_only=False):
    if not 0 <= max_review_fraction < 1:
        raise ValueError('max-review-fraction must be in [0,1)')
    if lineage['smoke'] or lineage['manifest_sha256'] != digest(ROOT / 'manifests/development_manifest.csv'):
        raise ValueError('Calibration requires full frozen DEVELOPMENT outputs')
    if any(i['record']['partition'] != 'development' for i in items):
        raise ValueError('Protected/evaluation records cannot calibrate thresholds')
    positive = [i for i in items if i['record']['image_ground_truth'] == 'valid' and decide(i['observation'], 'F1')['outcome'] == 'ACCEPT']
    negative = [i for i in items if i['record']['image_ground_truth'] == 'invalid_talim_content']
    if not allow_positive_only:
        return None, dict(status='NOT_CALIBRATED', reason='No justified content-rejection calibration experiment for current one-family fixtures',
                          positive_f1_accepted=len(positive), invalid_content_examples=len(negative),
                          next_action='Acquire independent, image-adjudicated invalid development parents, review/freeze a new protocol. Or explicitly choose positive-only quality gating.')
    values = sorted(float(i['observation']['confidence']['nonblank_geomean']) for i in positive
                    if i['observation']['confidence'].get('nonblank_geomean') is not None)
    if len(values) < 20 or not all(math.isfinite(v) and 0 <= v <= 1 for v in values):
        raise ValueError('Need at least 20 F1-accepted positive development pages with finite quality scores')
    threshold = values[min(int(len(values) * max_review_fraction), len(values) - 1)]
    policy = dict(schema_version=1, enabled=True, action='REVIEW_UNCERTAIN',
        minimum=dict(nonblank_geomean=threshold, ctc_log_score_per_frame=None, mean_top2_log_margin=None, role_margin=None),
        calibrated_validity_probability=False, scope='positive_only_quality_gate; NOT invalid-content calibrated',
        lineage=dict(development_manifest_sha256=lineage['manifest_sha256'],
                     observations_sha256=lineage['observations_sha256'], checkpoint_sha256=CHECKPOINT_SHA,
                     parents=sorted({i['record']['source_parent_id'] for i in items}),
                     f1_accepted_positive_pages=len(positive), max_additional_review_fraction=max_review_fraction))
    return policy, dict(status='QUALITY_GATE_ONLY', threshold=threshold, calibration_values=len(values),
                        observed_additional_review_fraction=sum(v < threshold for v in values) / len(values),
                        negative_recall_optimized=False, independent_test=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--baseline-dir', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--max-review-fraction', type=float, default=0.05)
    ap.add_argument('--allow-positive-only', action='store_true')
    args = ap.parse_args()
    items, lineage = load_baseline(args.baseline_dir)
    policy, report = calibrate(items, lineage, args.max_review_fraction, args.allow_positive_only)
    out = new_output(args.output_dir)
    write_json(out / 'calibration_report.json', report)
    if policy is not None:
        write_json(out / 'confidence_policy.yaml', policy)  # JSON is valid YAML 1.2
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
