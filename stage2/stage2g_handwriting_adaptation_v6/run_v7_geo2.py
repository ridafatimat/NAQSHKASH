"""Validation: geometry coverage extensions on top of the current best
(V7 + contrast + oval + shape + geometry + marks).

Rows compared (all on VALIDATION writers W05, W24, W30, W31 only):
  A  current best                       (must reproduce 15.7% macro / 53.8% exact)
  B  A + uniform centres                 (count tier: geometric glyph count != CRNN length)
  C  A + classifier read                 (mark tier: #ink slots != #CRNN marks)
  D  A + uniform + classifier read
Pre-declared rule (written before running): a variant is adopted only if its macro
CER is LOWER than A and its exact accuracy is NOT LOWER than A. If several qualify,
take the lowest macro CER. Run once; do not tweak settings after seeing results.

Usage:  python -B run_v7_geo2.py
"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_v6 as R          # noqa: E402
import run_v6_post as PP    # noqa: E402
import geometry2 as G2      # noqa: E402
from oval_fusion import OvalFixer    # noqa: E402
from mark_fusion import MarkFixer    # noqa: E402
from shape_fusion import ShapeFixer  # noqa: E402
e = R.e
V7 = HERE / 'outputs/v7_run/training/checkpoints/step_01750.pt'


def main():
    tr, val, _, _ = R.load_inputs(); m, v, _ = e.v2.load_model(V7); m.eval()
    ov = OvalFixer(HERE / 'oval_classifier.pt'); mf = MarkFixer(HERE / 'mark_classifier.pt'); sf = ShapeFixer(HERE / 'shape_classifier.pt', ov)
    prob, base = {}, {}
    for r in val:
        p = PP.probs(m, PP.variants(e.v2.tensors(r))['contrast']); prob[r['id']] = p; base[r['id']] = e.h.decode(p, v, r['role'], True)
        if r['role'] == 'count': base[r['id']] = sf.fix(ov.fix(base[r['id']], r['path'])[0], r['path'])[0]
    variants = [('A current best', False, False), ('B + uniform centres', True, False), ('C + classifier read', False, True), ('D + uniform + classifier read', True, True)]
    res = {}
    for name, uni, cr in variants:
        P, log = G2.place(val, base, mf, uniform=uni, classread=cr); rows = []
        for r in val:
            row = e.h.result_row(r, P[r['id']], prob[r['id']], v); row['writer_id'] = r['writer_id']; rows.append(row)
        met = e.all_metrics(rows); e.write(HERE / 'outputs' / ('geo2_' + name.split()[0] + '.csv'), rows)
        res[name] = dict(macro=met['macro_writer']['cer'], overall=met['overall']['cer'], exact=met['overall']['exact_accuracy'], le1=sum(int(x['edit_distance']) <= 1 for x in rows) / len(rows),
                         count=met['per_role']['count']['cer'], upper=met['per_role']['upper']['cer'], lower=met['per_role']['lower']['cer'],
                         per_writer={w: round(x['cer'] * 100, 1) for w, x in met['per_writer'].items()}, log=dict(log), uniform=uni, classread=cr)
    print('| Setup | Macro CER | Overall CER | Exact | <=1 error | Count | Upper | Lower |'); print('|---|---:|---:|---:|---:|---:|---:|---:|')
    for n, x in res.items(): print(f"| {n} | {x['macro']*100:.1f}% | {x['overall']*100:.1f}% | {x['exact']*100:.1f}% | {x['le1']*100:.1f}% | {x['count']*100:.1f}% | {x['upper']*100:.1f}% | {x['lower']*100:.1f}% |")
    for n, x in res.items(): print(n, x['per_writer'], x['log'])
    A = res['A current best']; ok = [(x['macro'], n) for n, x in res.items() if n != 'A current best' and x['macro'] < A['macro'] and x['exact'] >= A['exact']]
    choice = min(ok)[1] if ok else 'A current best'
    flags = {'A current best': '', 'B + uniform centres': ' --geo-uniform', 'C + classifier read': ' --geo-classread', 'D + uniform + classifier read': ' --geo-uniform --geo-classread'}[choice]
    print('\nDECISION (pre-declared rule):', choice)
    print('Freeze command:  python -B run_test_once.py --freeze --oval --shape --marks' + flags)
    res['decision'] = choice; e.dump(HERE / 'outputs' / 'v7_geo2_summary.json', res)


if __name__ == '__main__':
    main()
