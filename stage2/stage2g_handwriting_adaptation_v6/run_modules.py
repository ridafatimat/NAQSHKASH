"""Validation of two new post-processing modules on the CURRENT final model (V7 step 1750,
old classifiers). Same 65 validation tiers (W05 W24 W30 W31). No training.

  A  current final (contrast + oval + shape + geometry uniform + marks)   must give 15.1% / 53.8%
  B  A + count box reader, strict  (all glyphs p >= 0.6)
  C  A + count box reader, argmax
  D  A + mark fusion (CRNN x mark classifier)
  E  A + the adopted ones of {B or C} and D together
Pre-declared rule: B, C, D are each compared with A. A module qualifies if its macro CER is
LOWER than A and its exact accuracy is NOT LOWER. Between B and C, the qualifying one with the
lower macro CER is taken. E combines everything that qualified and is adopted only if it ALSO
beats A by the same rule; otherwise the single best qualifying module is used.

Usage (v6 folder):  python -B run_modules.py
"""
import sys, collections
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_v6 as R          # noqa: E402
import run_v6_post as PP    # noqa: E402
import geometry2 as G2      # noqa: E402
from oval_fusion import OvalFixer    # noqa: E402
from mark_fusion import MarkFixer    # noqa: E402
from shape_fusion import ShapeFixer  # noqa: E402
from count_reader import CountReader          # noqa: E402
from mark_fuse import FusedMarkFixer, crnn_mark_dists  # noqa: E402
e = R.e
V7 = HERE / 'outputs/v7_run/training/checkpoints/step_01750.pt'


def run(val, prob, v, base, ov, sf, reader_mode=None, fuse=False):
    pred = dict(base); log = collections.Counter()
    if reader_mode:
        cr = CountReader(sf, ov, reader_mode)
        for r in val:
            if r['role'] == 'count':
                pred[r['id']], n = cr.fix(pred[r['id']], r['path']); log['count_reader_applied'] += n
    if fuse:
        mf = FusedMarkFixer(HERE / 'mark_classifier.pt')
        for r in val:
            if r['role'] in ('upper', 'lower'):
                chars, dists = crnn_mark_dists(prob[r['id']], v, r['role'], set(e.h.allowed(r['role'])))
                mf.set_context(r['path'], chars, dists)
    else:
        mf = MarkFixer(HERE / 'mark_classifier.pt')
    pred, glog = G2.place(val, pred, mf, uniform=True); log.update(glog)
    if fuse: log['fusion_used'] = mf.used; log['fusion_fallback'] = mf.fallback
    rows = []
    for r in val:
        x = e.h.result_row(r, pred[r['id']], prob[r['id']], v); x['writer_id'] = r['writer_id']; rows.append(x)
    met = e.all_metrics(rows)
    return rows, dict(macro=met['macro_writer']['cer'], overall=met['overall']['cer'], exact=met['overall']['exact_accuracy'], le1=sum(int(x['edit_distance']) <= 1 for x in rows) / len(rows),
                      count=met['per_role']['count']['cer'], upper=met['per_role']['upper']['cer'], lower=met['per_role']['lower']['cer'],
                      per_writer={w: round(x['cer'] * 100, 1) for w, x in met['per_writer'].items()}, log=dict(log))


def better(x, a): return x['macro'] < a['macro'] and x['exact'] >= a['exact']


def main():
    tr, val, _, _ = R.load_inputs(); m, v, _ = e.v2.load_model(V7); m.eval()
    ov = OvalFixer(HERE / 'oval_classifier.pt'); sf = ShapeFixer(HERE / 'shape_classifier.pt', ov)
    prob, base = {}, {}
    for r in val:
        p = PP.probs(m, PP.variants(e.v2.tensors(r))['contrast']); prob[r['id']] = p; base[r['id']] = e.h.decode(p, v, r['role'], True)
        if r['role'] == 'count': base[r['id']] = sf.fix(ov.fix(base[r['id']], r['path'])[0], r['path'])[0]
    res = {}
    for name, rm, fu in [('A current final', None, False), ('B + count reader (strict)', 'strict', False), ('C + count reader (argmax)', 'argmax', False), ('D + mark fusion', None, True)]:
        rows, res[name] = run(val, prob, v, base, ov, sf, rm, fu); e.write(HERE / 'outputs' / ('modules_' + name.split()[0] + '.csv'), rows)
    A = res['A current final']
    reader = [n for n in ('B + count reader (strict)', 'C + count reader (argmax)') if better(res[n], A)]
    reader = min(reader, key=lambda n: res[n]['macro']) if reader else None
    fuse = better(res['D + mark fusion'], A)
    singles = ([reader] if reader else []) + (['D + mark fusion'] if fuse else [])
    choice = 'A current final'
    if len(singles) == 2:
        rows, res['E combined'] = run(val, prob, v, base, ov, sf, 'strict' if reader.startswith('B') else 'argmax', True)
        e.write(HERE / 'outputs' / 'modules_E.csv', rows)
        choice = 'E combined' if better(res['E combined'], A) else min(singles, key=lambda n: res[n]['macro'])
    elif singles: choice = singles[0]
    print('| Setup | Macro CER | Overall CER | Exact | <=1 error | Count | Upper | Lower |'); print('|---|---:|---:|---:|---:|---:|---:|---:|')
    for n, x in res.items(): print(f"| {n} | {x['macro']*100:.1f}% | {x['overall']*100:.1f}% | {x['exact']*100:.1f}% | {x['le1']*100:.1f}% | {x['count']*100:.1f}% | {x['upper']*100:.1f}% | {x['lower']*100:.1f}% |")
    for n, x in res.items(): print(n, x['per_writer'], x['log'])
    print('\nDECISION (pre-declared rule):', choice)
    res['decision'] = choice; e.dump(HERE / 'outputs' / 'modules_summary.json', res)


if __name__ == '__main__':
    main()
