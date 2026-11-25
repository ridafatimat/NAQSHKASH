"""Validation: full pipeline (contrast + role mask + oval + shape + count reader (argmax) + geometry(uniform) + marks (fusion))
with V7 vs V8 and old vs v2 classifiers. Needs outputs/v8_run (run_v8.py) and, for rows C/D,
oval_classifier_v2.pt, shape_classifier_v2.pt, mark_classifier_v2.pt.

  A  V7 + old classifiers   (current final; must reproduce 7.9% macro / 66.2% exact)
  B  V8 + old classifiers
  C  V7 + v2 classifiers
  D  V8 + v2 classifiers
Pre-declared rule: among B/C/D, those with macro CER LOWER than A and exact NOT LOWER than A
qualify; the lowest macro CER wins; otherwise A stays. Selection uses the same 65 validation
tiers (W05 W24 W30 W31) as before. W09 (3 recovered tiers) is reported separately only.

Usage:  python -B run_v8_final.py
"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_v6 as R           # noqa: E402
import inputs_v2             # noqa: E402
import run_v6_post as PP     # noqa: E402
import geometry2 as G2       # noqa: E402
from oval_fusion import OvalFixer    # noqa: E402
from mark_fusion import MarkFixer    # noqa: E402
from shape_fusion import ShapeFixer  # noqa: E402
from count_reader import CountReader          # noqa: E402
from mark_fuse import FusedMarkFixer, crnn_mark_dists  # noqa: E402
e = R.e
inputs_v2.use(R)
CK = {'V7': json.loads((HERE / 'outputs/v7_run/training/completion.json').read_text())['best_checkpoint'],
      'V8': json.loads((HERE / 'outputs/v8_run/training/completion.json').read_text())['best_checkpoint']}


def pipeline(ckpt, suffix, rows):
    m, v, _ = e.v2.load_model(ckpt); m.eval()
    ov = OvalFixer(HERE / f'oval_classifier{suffix}.pt'); sf = ShapeFixer(HERE / f'shape_classifier{suffix}.pt', ov); mf = FusedMarkFixer(HERE / f'mark_classifier{suffix}.pt'); cr = CountReader(sf, ov, 'argmax')
    prob, pred = {}, {}
    for r in rows:
        p = PP.probs(m, PP.variants(e.v2.tensors(r))['contrast'])   # identical to run_v7_geo2 / run_test_once
        prob[r['id']] = p; pred[r['id']] = e.h.decode(p, v, r['role'], True)
        if r['role'] == 'count': pred[r['id']] = cr.fix(sf.fix(ov.fix(pred[r['id']], r['path'])[0], r['path'])[0], r['path'])[0]
        else: mf.set_context(r['path'], *crnn_mark_dists(p, v, r['role'], set(e.h.allowed(r['role']))))
    pred, log = G2.place(rows, pred, mf, uniform=True)
    out = []
    for r in rows:
        x = e.h.result_row(r, pred[r['id']], prob[r['id']], v); x['writer_id'] = r['writer_id']; out.append(x)
    return out, dict(log)


def summ(rows):
    met = e.all_metrics(rows)
    return dict(macro=met['macro_writer']['cer'], overall=met['overall']['cer'], exact=met['overall']['exact_accuracy'], le1=sum(int(x['edit_distance']) <= 1 for x in rows) / len(rows),
                count=met['per_role']['count']['cer'], upper=met['per_role'].get('upper', {}).get('cer', float('nan')), lower=met['per_role'].get('lower', {}).get('cer', float('nan')),
                per_writer={w: round(x['cer'] * 100, 1) for w, x in met['per_writer'].items()})


def main():
    tr, val, _, _ = R.load_inputs(); extra = R.load_extra()
    have_v2 = all((HERE / f'{k}_classifier_v2.pt').exists() for k in ('oval', 'shape', 'mark'))
    setups = [('A V7 + old classifiers', 'V7', ''), ('B V8 + old classifiers', 'V8', '')]
    if have_v2: setups += [('C V7 + v2 classifiers', 'V7', '_v2'), ('D V8 + v2 classifiers', 'V8', '_v2')]
    else: print('NOTE: v2 classifiers not found; only A and B are compared')
    res = {}
    for name, ck, sfx in setups:
        rows, log = pipeline(CK[ck], sfx, val); xrows, _ = pipeline(CK[ck], sfx, extra)
        e.write(HERE / 'outputs' / ('v8_final_' + name.split()[0] + '.csv'), rows); e.write(HERE / 'outputs' / ('v8_final_' + name.split()[0] + '_W09.csv'), xrows)
        res[name] = dict(val=summ(rows), W09=summ(xrows), log=log, checkpoint=CK[ck], classifiers=sfx or '_v1')
    print('| Setup (validation: 65 tiers, W05 W24 W30 W31) | Macro CER | Overall CER | Exact | <=1 error | Count | Upper | Lower | W09 extra (3 tiers) CER |')
    print('|---|---:|---:|---:|---:|---:|---:|---:|---:|')
    for n, x in res.items():
        a = x['val']; print(f"| {n} | {a['macro']*100:.1f}% | {a['overall']*100:.1f}% | {a['exact']*100:.1f}% | {a['le1']*100:.1f}% | {a['count']*100:.1f}% | {a['upper']*100:.1f}% | {a['lower']*100:.1f}% | {x['W09']['overall']*100:.1f}% |")
    for n, x in res.items(): print(n, x['val']['per_writer'], x['log'])
    A = res['A V7 + old classifiers']['val']
    ok = [(x['val']['macro'], n) for n, x in res.items() if not n.startswith('A') and x['val']['macro'] < A['macro'] and x['val']['exact'] >= A['exact']]
    choice = min(ok)[1] if ok else 'A V7 + old classifiers'; c = res[choice]
    print('\nDECISION (pre-declared rule):', choice)
    flags = '--count-reader argmax --mark-fusion ' + f'--checkpoint "{c["checkpoint"]}" --label "{choice[2:4]}"' + (' --classifiers v2' if c['classifiers'] == '_v2' else '')
    print('Freeze command:  python -B run_test_once.py --freeze --oval --shape --marks --geo-uniform ' + flags)
    res['decision'] = choice; e.dump(HERE / 'outputs' / 'v8_final_summary.json', res)


if __name__ == '__main__':
    main()
