"""Validation: V7 + contrast + oval + geometry, with and without the mark classifier.
Needs oval_classifier.pt and mark_classifier.pt in this folder.
Usage:  python -B run_v7_marks.py
"""
import sys, collections
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_v6 as R          # noqa: E402
import run_v6_post as PP    # noqa: E402
import run_v6_geo as G      # noqa: E402
from oval_fusion import OvalFixer   # noqa: E402
from mark_fusion import MarkFixer   # noqa: E402
e = R.e
V7 = HERE / 'outputs/v7_run/training/checkpoints/step_01750.pt'


def geometry(rows, pred, marks=None):
    """Geometry placement; optionally re-read every placed mark with the mark classifier."""
    pred = dict(pred); by = collections.defaultdict(dict); log = collections.Counter()
    for r in rows: by[(r['source_id'], r['row_index'])][r['role']] = r
    for g in by.values():
        if 'count' not in g: continue
        cen, Wc = G.glyph_centres(g['count']['path'])
        if len(cen) != len(pred[g['count']['id']]): continue
        for role in ('upper', 'lower'):
            if role not in g: continue
            r = g[role]; mk = pred[r['id']].replace(' ', ''); slots, Wm = G.mark_slots(r['path'], cen)
            if abs(Wm - Wc) > G.WIDTH_TOL * Wc or not mk or len(slots) != len(mk): continue
            s = [' '] * (max(slots) + 1)
            for ch, j in zip(mk, slots): s[j] = ch
            placed = ''.join(s)
            if marks is not None:
                placed, n = marks.fix(placed, role, r['path'], g['count']['path'], cen); log['marks_changed'] += n; log['tiers_checked'] += 1
            pred[r['id']] = placed.strip(' ')
    return pred, log


def main():
    tr, val, _, _ = R.load_inputs(); m, v, _ = e.v2.load_model(V7); m.eval()
    ov = OvalFixer(HERE / 'oval_classifier.pt'); mf = MarkFixer(HERE / 'mark_classifier.pt')
    prob, base = {}, {}
    for r in val:
        p = PP.probs(m, PP.variants(e.v2.tensors(r))['contrast']); prob[r['id']] = p; base[r['id']] = e.h.decode(p, v, r['role'], True)
        if r['role'] == 'count': base[r['id']] = ov.fix(base[r['id']], r['path'])[0]
    res = {}
    for name, mk in [('V7+contrast+oval+geometry', None), ('V7+contrast+oval+geometry+marks', mf)]:
        P, log = geometry(val, base, mk); rows = []
        for r in val:
            row = e.h.result_row(r, P[r['id']], prob[r['id']], v); row['writer_id'] = r['writer_id']; rows.append(row)
        met = e.all_metrics(rows); e.write(HERE / 'outputs' / f'{name}.csv', rows)
        res[name] = dict(macro=met['macro_writer']['cer'], overall=met['overall']['cer'], exact=met['overall']['exact_accuracy'], le1=sum(int(x['edit_distance']) <= 1 for x in rows) / len(rows),
                         upper=met['per_role']['upper']['cer'], lower=met['per_role']['lower']['cer'], per_writer={w: round(x['cer'] * 100, 1) for w, x in met['per_writer'].items()}, log=dict(log))
    print('| Setup | Macro CER | Overall CER | Exact | <=1 error | Upper | Lower |'); print('|---|---:|---:|---:|---:|---:|---:|')
    for n, x in res.items(): print(f"| {n} | {x['macro']*100:.1f}% | {x['overall']*100:.1f}% | {x['exact']*100:.1f}% | {x['le1']*100:.1f}% | {x['upper']*100:.1f}% | {x['lower']*100:.1f}% |")
    print({n: (x['per_writer'], x['log']) for n, x in res.items()}); e.dump(HERE / 'outputs' / 'v7_marks_summary.json', res)


if __name__ == '__main__':
    main()
