"""Render saved comparison results. Does not synthesize or execute experiments."""
import argparse
from common import *


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--comparison', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if not args.output.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError('Reports must stay inside Stage2F')
    data = load_json(args.comparison)
    lines = ['# Stage 2F saved-results comparison', '',
        'Exposed development/regression and one negative fixture family. Not independent unseen evidence.', '',
        'REVIEW remains separate from REJECT. Null = absent denominator, not zero error.', '',
        '| Candidate | Invalid n | Reject recall | False acceptance | Positive accept | Positive reject | Positive review | Accepted exact |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    def show(v):
        return 'N/A' if v is None else f'{v:.4f}'
    for row in data['table']:
        lines.append('| ' + ' | '.join([row['candidate'], str(row['invalid_denominator'])] + [show(row[k]) for k in
            ('negative_rejection_recall', 'false_acceptance_rate', 'positive_acceptance_rate', 'false_rejection_rate',
             'positive_review_rate', 'accepted_output_correctness')]) + ' |')
    lines += ['', 'Invalid denominators in this fixture suite describe input artifacts, not demonstrated invalid Talim content.',
              'See candidate groups.json for category, parent, origin and image-ground-truth groups; summary.json includes parent macro rates.',
              'No learned classifier was justified. No F3 results exist.', '',
              'Source comparison SHA-256: ' + digest(args.comparison)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(args.output)


if __name__ == '__main__':
    main()
