"""Validation: V7 + contrast + geometry, with and without the oval classifier.
Usage (v6 folder, oval_classifier.pt next to this file):  python -B run_v7_oval.py
"""
import sys, json, collections
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_v6 as R          # noqa: E402
import run_v6_post as PP    # noqa: E402
import run_v6_geo as G      # noqa: E402
from oval_fusion import OvalFixer  # noqa: E402
e = R.e
V7 = HERE / 'outputs/v7_run/training/checkpoints/step_01750.pt'


def geometry(rows, pred):
    pred = dict(pred); by = collections.defaultdict(dict)
    for r in rows: by[(r['source_id'], r['row_index'])][r['role']] = r
    for g in by.values():
        if 'count' not in g: continue
        cen, Wc = G.glyph_centres(g['count']['path'])
        if len(cen) != len(pred[g['count']['id']]): continue
        for role in ('upper', 'lower'):
            if role not in g: continue
            r = g[role]; marks = pred[r['id']].replace(' ', ''); slots, Wm = G.mark_slots(r['path'], cen)
            if abs(Wm - Wc) > G.WIDTH_TOL * Wc or not marks or len(slots) != len(marks): continue
            s = [' '] * (max(slots) + 1)
            for ch, j in zip(marks, slots): s[j] = ch
            pred[r['id']] = ''.join(s).strip(' ')
    return pred


def main():
    tr, val, _, dev = R.load_inputs(); m, v, _ = e.v2.load_model(V7); m.eval(); fx = OvalFixer(HERE / 'oval_classifier.pt')
    prob, base = {}, {}
    for r in val:
        p = PP.probs(m, PP.variants(e.v2.tensors(r))['contrast']); prob[r['id']] = p; base[r['id']] = e.h.decode(p, v, r['role'], True)
    ov = dict(base); log = collections.Counter()
    for r in val:
        if r['role'] != 'count': continue
        new, n, applied = fx.fix(base[r['id']], r['path']); ov[r['id']] = new; log['applied'] += applied; log['glyphs_changed'] += n
    res = {}
    for name, P in [('V7+contrast+geometry', geometry(val, base)), ('V7+contrast+oval+geometry', geometry(val, ov))]:
        rows = []
        for r in val:
            row = e.h.result_row(r, P[r['id']], prob[r['id']], v); row['writer_id'] = r['writer_id']; rows.append(row)
        met = e.all_metrics(rows); e.write(HERE / 'outputs' / f'{name}.csv', rows)
        res[name] = dict(macro=met['macro_writer']['cer'], overall=met['overall']['cer'], exact=met['overall']['exact_accuracy'], le1=sum(int(x['edit_distance']) <= 1 for x in rows) / len(rows),
                         count=met['per_role']['count']['cer'], per_writer={w: round(x['cer'] * 100, 1) for w, x in met['per_writer'].items()})
    print('oval classifier:', dict(log))
    print('| Setup | Macro CER | Overall CER | Exact | <=1 error | Count CER |'); print('|---|---:|---:|---:|---:|---:|')
    for n, x in res.items(): print(f"| {n} | {x['macro']*100:.1f}% | {x['overall']*100:.1f}% | {x['exact']*100:.1f}% | {x['le1']*100:.1f}% | {x['count']*100:.1f}% |")
    print({n: x['per_writer'] for n, x in res.items()}); e.dump(HERE / 'outputs' / 'v7_oval_summary.json', dict(results=res, log=dict(log)))


if __name__ == '__main__':
    main()
