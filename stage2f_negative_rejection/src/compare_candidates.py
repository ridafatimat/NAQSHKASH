"""Compare actual completed F0/F1/F2 outputs with paired coverage and lineage."""
import argparse
from common import *


def compare(directories):
    results = []
    for directory in directories:
        p = Path(directory)
        done = load_json(p / 'completion.json')
        if not done.get('complete') or digest(p / 'results.json') != done['results_sha256']:
            raise ValueError('Incomplete/drifted candidate: ' + str(p))
        result = load_json(p / 'results.json')
        if len(result['rows']) != done['rows']:
            raise ValueError('Candidate row count mismatch')
        results.append(result)
    if not results or len({r['candidate'] for r in results}) != len(results):
        raise ValueError('Provide one completed directory per distinct candidate')
    first = results[0]
    coverage = {(r['image_id'], r['source_parent_id']) for r in first['rows']}
    for r in results:
        if {(x['image_id'], x['source_parent_id']) for x in r['rows']} != coverage or len(r['rows']) != len(coverage):
            raise ValueError('Candidate coverage differs or contains duplicates')
        for field in ('manifest_sha256', 'checkpoint_sha256', 'observations_sha256'):
            if r['lineage'][field] != first['lineage'][field]:
                raise ValueError('Unpaired candidate lineage: ' + field)
        if r['lineage']['smoke']:
            raise ValueError('Cannot compare smoke as full experiment')
    from evaluate_rejection import metrics
    table = []
    for r in results:
        m = metrics(r['rows'])
        table.append(dict(candidate=r['candidate'], **{k: v for k, v in m.items() if not isinstance(v, dict)}))
    return table, results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--candidate-dir', action='append', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    args = ap.parse_args()
    table, results = compare(args.candidate_dir)
    out = new_output(args.output_dir)
    write_csv(out / 'comparison.csv', table)
    write_json(out / 'comparison.json', dict(table=table, results=results, independent_test=False))
    print('Comparison saved:', out)


if __name__ == '__main__':
    main()
