"""Apply F0/F1/F2 to saved, complete baseline outputs; no model execution."""
import argparse
from collections import Counter, defaultdict
from common import *
from rejection_pipeline import decide, load_decision_policy, DEFAULT_DECISION_POLICY

VALID = {'valid'}
INVALID = {'corrupted_unreadable', 'empty_input', 'invalid_talim_content'}
OUTCOMES = ('ACCEPT', 'REJECT_INVALID', 'REVIEW_UNCERTAIN')


def ratio(n, d):
    return n / d if d else None


def metrics(rows):
    pos = [r for r in rows if r['image_ground_truth'] in VALID]
    neg = [r for r in rows if r['image_ground_truth'] in INVALID]
    accepted_scorable = [r for r in pos if r['outcome'] == 'ACCEPT' and r['transcription_exact'] is not None]
    def count(group, outcome):
        return sum(r['outcome'] == outcome for r in group)
    content = [r for r in neg if r['image_ground_truth'] == 'invalid_talim_content']
    scorable = [r for r in rows if r['expected_system_outcome'] != 'ACCEPT' or r['transcription_exact'] is not None]
    return dict(total=len(rows), valid_denominator=len(pos), invalid_denominator=len(neg),
        invalid_content_denominator=len(content), other_denominator=len(rows) - len(pos) - len(neg),
        outcome_counts=dict(Counter(r['outcome'] for r in rows)),
        negative_rejection_recall=ratio(count(neg, 'REJECT_INVALID'), len(neg)),
        false_acceptance_rate=ratio(count(neg, 'ACCEPT'), len(neg)),
        invalid_review_rate=ratio(count(neg, 'REVIEW_UNCERTAIN'), len(neg)),
        content_rejection_recall=ratio(count(content, 'REJECT_INVALID'), len(content)),
        positive_acceptance_rate=ratio(count(pos, 'ACCEPT'), len(pos)),
        false_rejection_rate=ratio(count(pos, 'REJECT_INVALID'), len(pos)),
        positive_review_rate=ratio(count(pos, 'REVIEW_UNCERTAIN'), len(pos)),
        positive_nonacceptance_rate=ratio(len(pos) - count(pos, 'ACCEPT'), len(pos)),
        accepted_output_correctness=ratio(sum(r['transcription_exact'] for r in accepted_scorable), len(accepted_scorable)),
        accepted_output_denominator=len(accepted_scorable),
        end_to_end_page_correctness=ratio(sum(r['outcome'] == r['expected_system_outcome'] and
            (r['expected_system_outcome'] != 'ACCEPT' or r['transcription_exact'] is True) for r in scorable), len(scorable)),
        end_to_end_denominator=len(scorable),
        expected_action_accuracy=ratio(sum(r['outcome'] == r['expected_system_outcome'] for r in rows), len(rows)),
        rejection_reason_accuracy=None, rejection_reason_denominator=0)


def load_baseline(directory, allow_smoke=False):
    directory = Path(directory)
    completion = load_json(directory / 'completion.json')
    if not completion['complete'] or completion['checkpoint_sha256'] != CHECKPOINT_SHA:
        raise ValueError('Incomplete or incompatible baseline run')
    if completion['smoke'] and not allow_smoke:
        raise ValueError('Smoke outputs cannot enter full experiment results')
    path = directory / 'observations.jsonl'
    if digest(path) != completion['observations_sha256']:
        raise ValueError('Observation output drift')
    manifest = Path(completion['manifest'])
    if manifest.resolve() not in {(ROOT / f'manifests/{part}_manifest.csv').resolve() for part in ('development', 'evaluation')}:
        raise ValueError('Unrecognized manifest lineage')
    if digest(manifest) != completion['manifest_sha256']:
        raise ValueError('Manifest lineage drift')
    items = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    if len(items) != completion['rows']:
        raise ValueError('Row count mismatch')
    expected = {r['image_id']: r for r in read_csv(manifest)}
    if not completion['smoke'] and {i['record']['image_id'] for i in items} != set(expected):
        raise ValueError('Incomplete image coverage')
    ids = [i['record']['image_id'] for i in items]
    if len(ids) != len(set(ids)) or any(i['record'] != expected[i['record']['image_id']] for i in items):
        raise ValueError('Record duplication or annotation drift')
    return items, completion


def evaluate(items, candidate, policy=None, decision_policy=None):
    rows = []
    for item in items:
        record = item['record']
        decision = decide(item['observation'], candidate, policy, decision_policy)
        rows.append(dict(image_id=record['image_id'], source_parent_id=record['source_parent_id'],
            invalidity_category=record['invalidity_category'], data_origin=record['data_origin'],
            image_ground_truth=record['image_ground_truth'], expected_system_outcome=record['expected_system_outcome'],
            transcription_exact=item['transcription_exact'], **decision))
    groups = {}
    for field in ('invalidity_category', 'source_parent_id', 'data_origin', 'image_ground_truth'):
        groups[field] = {key: metrics([r for r in rows if r[field] == key]) for key in sorted({r[field] for r in rows})}
    summary = metrics(rows)
    # Parent macro avoids letting augmentations determine all weights.
    parent_metrics = list(groups['source_parent_id'].values())
    summary['macro_parent'] = {key: (sum(values) / len(values) if values else None)
        for key in ('negative_rejection_recall', 'false_acceptance_rate', 'positive_acceptance_rate', 'false_rejection_rate')
        for values in [[m[key] for m in parent_metrics if m[key] is not None]]}
    return rows, summary, groups


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--baseline-dir', type=Path, required=True)
    ap.add_argument('--candidate', choices=('F0', 'F1', 'F2'), required=True)
    ap.add_argument('--policy', type=Path)
    ap.add_argument('--decision-policy', type=Path, default=DEFAULT_DECISION_POLICY,
                    help='Versioned F1/F2 evidence-to-action mapping; separate from F2 confidence --policy')
    ap.add_argument('--output-dir', type=Path, required=True)
    args = ap.parse_args()
    items, lineage = load_baseline(args.baseline_dir)
    policy = load_json(args.policy) if args.policy else None
    decision_policy = load_decision_policy(args.decision_policy) if args.candidate != 'F0' else None
    if args.candidate == 'F2' and policy and policy.get('enabled'):
        calibration = policy.get('lineage', {})
        if calibration.get('development_manifest_sha256') != digest(ROOT / 'manifests/development_manifest.csv'):
            raise ValueError('F2 enabled policy lacks frozen development provenance')
        if {i['record']['source_parent_id'] for i in items if i['record']['partition'] == 'evaluation'} & set(calibration.get('parents', [])):
            raise ValueError('Calibration/evaluation parent overlap')
    rows, summary, groups = evaluate(items, args.candidate, policy, decision_policy)
    out = new_output(args.output_dir)
    write_json(out / 'results.json', dict(candidate=args.candidate, lineage=lineage, policy=policy, rows=rows,
        decision_policy=decision_policy, decision_policy_path=str(args.decision_policy) if decision_policy else None,
        decision_policy_sha256=digest(args.decision_policy) if decision_policy else None))
    write_json(out / 'summary.json', summary)
    write_json(out / 'groups.json', groups)
    csvrows = [dict(r, reason_codes=';'.join(r['reason_codes'])) for r in rows]
    for row in csvrows:
        for key, value in list(row.items()):
            if isinstance(value, (dict, list)):
                row[key] = json.dumps(value, ensure_ascii=False)
    write_csv(out / 'per_image.csv', csvrows, sorted({k for r in csvrows for k in r}))
    # Normalize optional decision evidence for heterogeneous rows.
    write_json(out / 'completion.json', dict(complete=True, candidate=args.candidate, rows=len(rows), results_sha256=digest(out / 'results.json')))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
