"""NAQSHKASH protected test: run ONCE on the frozen final setup.

Test writers: W02, W07 (human-verified crops, writer split v2 test manifest)
              + W14 W17 W20 W25 W27 W28 (collection v3; only tiers a teammate
              marked qa=OK in test_tier_crops/manifest.csv).
Models:       frozen E1 (raw input), V5b final (raw input) as baselines, and the
              FINAL setup frozen in final_setup.json (contrast + role mask + geometry).
The script refuses to run twice and refuses if final_setup.json changed.

Usage (v6 folder):
  python -B run_test_once.py --freeze      writes final_setup.json (do this BEFORE testing)
  python -B run_test_once.py               runs the protected test once
"""
import sys, json, csv, io, zipfile, argparse, time, collections, hashlib
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_v6 as R          # noqa: E402
import run_v6_post as PP    # noqa: E402
import run_v6_geo as G      # noqa: E402
import geometry2 as G2      # noqa: E402
e, cv, np = R.e, R.cv, R.np

DL = Path.home() / 'Downloads'
SPLIT_ZIP = DL / 'naqshkash_stage2g_writer_split_v2_final.zip'
NEW_DIR = HERE / 'test_tier_crops'
SETUP = HERE / 'final_setup.json'
OUT = HERE / 'outputs' / 'protected_test'
V5B = 'C:/NAQSHKASH/stage2/stage2g_handwriting_adaptation_v5b/local_run/outputs/final_1794380015655170100/training/checkpoints/attempt_001_step_02000.pt'
FINAL = 'C:/NAQSHKASH/stage2/stage2g_handwriting_adaptation_v6/outputs/v7_run/training/checkpoints/step_01750.pt'


def freeze(ckpt, label, oval, marks=False, shape=False, geo_uniform=False, geo_classread=False, sfx='', reader='none', fuse=False, alt=False):
    assert not SETUP.exists(), 'final_setup.json already exists; never re-freeze after looking at test results'
    s = dict(model=label, checkpoint=ckpt, checkpoint_sha256=e.sha(ckpt), input='contrast', decoder='role_mask', geometry_alignment=True,
             geometry_settings=dict(gap=G.GAP_FRAC, speck=G.SPECK_FRAC, width_tol=G.WIDTH_TOL), geo_uniform=geo_uniform, geo_classread=geo_classread, frozen_unix=time.time(),
             chosen_on='validation only (W05, W24, W30, W31); test writers never used')
    s['geometry2_sha256'] = e.sha(HERE / 'geometry2.py')
    assert reader == 'none' or (oval and shape), 'the count reader needs --oval and --shape'
    assert not fuse or marks, 'mark fusion needs --marks'
    s['count_reader'] = reader; s['mark_fusion'] = fuse
    if alt:   # report-only comparison: same frozen setup with the OTHER classifier set; never selectable
        osfx = '' if sfx == '_v2' else '_v2'; s['report_only_alt_classifiers'] = {}
        for k in ('oval', 'mark', 'shape'):
            f = HERE / f'{k}_classifier{osfx}.pt'; s['report_only_alt_classifiers'][k] = dict(path=str(f), sha256=e.sha(f))
    if reader != 'none': s['count_reader_sha256'] = e.sha(HERE / 'count_reader.py')
    if fuse: s['mark_fuse_sha256'] = e.sha(HERE / 'mark_fuse.py')
    if oval: s['oval_classifier'] = str(HERE / f'oval_classifier{sfx}.pt'); s['oval_classifier_sha256'] = e.sha(HERE / f'oval_classifier{sfx}.pt')
    if marks: s['mark_classifier'] = str(HERE / f'mark_classifier{sfx}.pt'); s['mark_classifier_sha256'] = e.sha(HERE / f'mark_classifier{sfx}.pt')
    if shape: s['shape_classifier'] = str(HERE / f'shape_classifier{sfx}.pt'); s['shape_classifier_sha256'] = e.sha(HERE / f'shape_classifier{sfx}.pt')
    e.dump(SETUP, s); e.dump(HERE / 'final_setup.lock.json', dict(sha256=e.sha(SETUP))); print('FROZEN', s)


def find_crop(pkg, crop_path, sha, zips):
    """Return the crop bytes whose sha256 equals the manifest value. Looks in the named ZIP,
    then in every Downloads ZIP with the same stem (Windows '(n)' copies included), then on disk under C:/NAQSHKASH."""
    stem = pkg[:-4].rstrip('_0123456789').rstrip(' ')
    cands = [DL / pkg] + sorted(p for p in DL.glob(stem + '*.zip') if p.name != pkg)
    base = Path(crop_path).name
    for zpath in cands:
        if not zpath.is_file(): continue
        try:
            if zpath not in zips: zips[zpath] = zipfile.ZipFile(zpath)
        except zipfile.BadZipFile: continue
        zp = zips[zpath]
        for n in zp.namelist():
            if n.endswith('/' + base) or n == base:
                d = zp.read(n)
                if hashlib.sha256(d).hexdigest() == sha: print('  crop', base, 'from', zpath.name, flush=True); return d
    for f in Path('C:/NAQSHKASH').rglob(base):
        d = f.read_bytes()
        if hashlib.sha256(d).hexdigest() == sha: print('  crop', base, 'from', f, flush=True); return d
    return None


def test_rows(allow_missing=False):
    rows = []; missing = []
    # W02 / W07 originals from the v2 split (human-verified crops)
    z = zipfile.ZipFile(SPLIT_ZIP); pre = 'naqshkash_stage2g_writer_split_v2_final/'
    man = list(csv.DictReader(io.StringIO(z.read(pre + 'test_manifest_v2.csv').decode('utf-8-sig'))))
    man = [r for r in man if r['sample_type'] == 'original']
    tmp = OUT / 'w02_w07_crops'; tmp.mkdir(parents=True, exist_ok=True); zips = {}
    for r in man:
        assert r['writer_id'] in ('W02', 'W07')
        pkg = r['package'].split(' (')[0].strip()                      # e.g. 'handwritten_tier_crops_5.zip (protected_test/, not opened)'
        data = find_crop(pkg, r['crop_path'], r['crop_sha256'], zips)
        if data is None:
            assert allow_missing, ('no file with the manifest sha256 found for', r['tier_id'], pkg, 'search: Downloads ZIPs + C:/NAQSHKASH; to exclude unavailable crops run with --allow-missing')
            missing.append(r['tier_id']); continue
        dest = tmp / Path(r['crop_path']).name; dest.write_bytes(data)
        rows.append(dict(id=r['sample_id'], tier_id=r['tier_id'], source_id=r['source_id'], writer_id=r['writer_id'], role=r['tier'], row_index=r['row_index'],
                         truth=r['ocr_text'], path=str(dest), group='W02_W07'))
    # new test writers, teammate-QA'd crops
    for r in csv.DictReader(open(NEW_DIR / 'manifest.csv', encoding='utf-8-sig')):
        if r.get('qa', '').strip().upper() != 'OK' or not r['crop_path']: continue
        rows.append(dict(id=r['tier_id'], tier_id=r['tier_id'], source_id=r['source_id'], writer_id=r['writer_id'], role=r['tier'], row_index=r['row_index'],
                         truth=r['raw_label'].lstrip(' '), path=str(NEW_DIR / r['crop_path']), group='collection_v3'))
    tr, val, _, _ = R.load_inputs()
    used = {r['writer_id'] for r in tr + val}; assert not used & {r['writer_id'] for r in rows}, 'test writer appears in train/validation'
    if (HERE / 'collection_v3_inputs_v2.csv').exists():
        used2 = {r['writer_id'] for r in csv.DictReader(open(HERE / 'collection_v3_inputs_v2.csv', encoding='utf-8'))}
        assert not used2 & {r['writer_id'] for r in rows}, 'test writer appears in inputs v2'
    if missing: print('EXCLUDED (crop file unavailable, decided before testing):', missing, flush=True)
    return rows, missing


def run_model(path, rows, contrast, geometry, oval=None, marks=None, shape=None, uniform=False, classread=False, reader='none', fuse=False):
    m, v, _ = e.v2.load_model(path); m.eval(); pred, prob = {}, {}
    for r in rows:
        t = e.v2.tensors(r); t = PP.variants(t)['contrast'] if contrast else t
        p = PP.probs(m, t); prob[r['id']] = p; pred[r['id']] = e.h.decode(p, v, r['role'], True)
    ofx = mfx = sfx = None; log = {}
    if oval:
        from oval_fusion import OvalFixer
        ofx = OvalFixer(oval)
        for r in rows:
            if r['role'] == 'count': pred[r['id']] = ofx.fix(pred[r['id']], r['path'])[0]
    if shape:
        from shape_fusion import ShapeFixer
        sfx = ShapeFixer(shape, ofx)
        for r in rows:
            if r['role'] == 'count': pred[r['id']] = sfx.fix(pred[r['id']], r['path'])[0]
    if reader != 'none':
        from count_reader import CountReader
        cr = CountReader(sfx, ofx, reader); n = 0
        for r in rows:
            if r['role'] == 'count': pred[r['id']], k = cr.fix(pred[r['id']], r['path']); n += k
        log['count_reader_applied'] = n
    if marks and fuse:
        from mark_fuse import FusedMarkFixer, crnn_mark_dists
        mfx = FusedMarkFixer(marks)
        for r in rows:
            if r['role'] in ('upper', 'lower'): mfx.set_context(r['path'], *crnn_mark_dists(prob[r['id']], v, r['role'], set(e.h.allowed(r['role']))))
    elif marks:
        from mark_fusion import MarkFixer
        mfx = MarkFixer(marks)
    if geometry:
        pred, glog = G2.place(rows, pred, mfx, uniform=uniform, classread=classread); log.update(glog)
    out = []
    for r in rows:
        row = e.h.result_row(r, pred[r['id']], prob[r['id']], v); row['writer_id'] = r['writer_id']; row['group'] = r['group']; out.append(row)
    return out, log


def summarise(rows):
    met = e.all_metrics(rows)
    return dict(tiers=len(rows), writers=len({r['writer_id'] for r in rows}), macro_writer_CER=met['macro_writer']['cer'], overall_CER=met['overall']['cer'],
                exact=met['overall']['exact_accuracy'], le1=sum(int(x['edit_distance']) <= 1 for x in rows) / len(rows),
                per_role={k: x['cer'] for k, x in met['per_role'].items()}, per_writer={k: x['cer'] for k, x in met['per_writer'].items()})


def main():
    a = argparse.ArgumentParser(); a.add_argument('--freeze', action='store_true'); a.add_argument('--checkpoint', default=FINAL); a.add_argument('--label', default='V7 step 1750'); a.add_argument('--oval', action='store_true'); a.add_argument('--marks', action='store_true'); a.add_argument('--shape', action='store_true'); a.add_argument('--geo-uniform', action='store_true'); a.add_argument('--geo-classread', action='store_true'); a.add_argument('--classifiers', choices=['v1', 'v2'], default='v1'); a.add_argument('--count-reader', choices=['none', 'strict', 'argmax'], default='none'); a.add_argument('--mark-fusion', action='store_true'); a.add_argument('--report-alt-classifiers', action='store_true'); a.add_argument('--allow-missing', action='store_true'); args = a.parse_args()
    if args.freeze: return freeze(args.checkpoint, args.label, args.oval, args.marks, args.shape, args.geo_uniform, args.geo_classread, '_v2' if args.classifiers == 'v2' else '', args.count_reader, args.mark_fusion, args.report_alt_classifiers)
    assert SETUP.exists(), 'run --freeze first'
    assert e.sha(SETUP) == json.loads((HERE / 'final_setup.lock.json').read_text())['sha256'], 'final_setup.json changed after freezing'
    setup = json.loads(SETUP.read_text()); assert e.sha(setup['checkpoint']) == setup['checkpoint_sha256']
    assert not (OUT / 'test_summary.json').exists(), 'protected test already run; results are final'
    OUT.mkdir(parents=True, exist_ok=True); rows, missing = test_rows(args.allow_missing)
    print('TEST tiers', len(rows), 'writers', sorted({r['writer_id'] for r in rows}), flush=True)
    grp = collections.Counter(r['group'] for r in rows); print('per group', dict(grp), flush=True)
    assert grp['W02_W07'] > 0, 'no W02/W07 rows: check the v2 split ZIP and package ZIPs in Downloads'
    assert grp['collection_v3'] > 0, 'no qa=OK rows in test_tier_crops/manifest.csv: finish the teammate QA first'
    man = list(csv.DictReader(open(NEW_DIR / 'manifest.csv', encoding='utf-8-sig')))
    with_ok = {r['writer_id'] for r in man if r.get('qa', '').strip().upper() == 'OK'}
    no_ok = sorted({r['writer_id'] for r in man} - with_ok)
    expected = {'W02', 'W07'} | with_ok; got = {r['writer_id'] for r in rows}
    assert got == expected and got <= set(R.TEST_WRITERS), ('test writers', sorted(got), 'expected', sorted(expected))
    print('test writers used', len(got), sorted(got), '| assigned test writers with no OK crop (excluded by QA):', no_ok, flush=True)
    res = {}
    if setup.get('geometry2_sha256'): assert e.sha(HERE / 'geometry2.py') == setup['geometry2_sha256'], 'geometry2.py changed after freezing'
    if setup.get('count_reader_sha256'): assert e.sha(HERE / 'count_reader.py') == setup['count_reader_sha256'], 'count_reader.py changed after freezing'
    if setup.get('mark_fuse_sha256'): assert e.sha(HERE / 'mark_fuse.py') == setup['mark_fuse_sha256'], 'mark_fuse.py changed after freezing'
    for k in ('oval_classifier', 'mark_classifier', 'shape_classifier'):
        if setup.get(k): assert e.sha(setup[k]) == setup[k + '_sha256'], k + ' changed after freezing'
    ov, mk, sh = setup.get('oval_classifier'), setup.get('mark_classifier'), setup.get('shape_classifier')
    gu, gc = bool(setup.get('geo_uniform')), bool(setup.get('geo_classread')); rd, fu = setup.get('count_reader', 'none'), bool(setup.get('mark_fusion'))
    final_name = ('FINAL: ' + setup['model'] + ' + contrast' + (' + oval' if ov else '') + (' + shape' if sh else '') + (f' + count reader ({rd})' if rd != 'none' else '')
                  + ' + geometry' + (' (uniform)' if gu else '') + (' + marks' if mk else '') + (' (fusion)' if fu else '') + (' (classifier read)' if gc else ''))
    F = dict(path=setup['checkpoint'], contrast=True, geometry=True, oval=ov, marks=mk, shape=sh, uniform=gu, classread=gc, reader=rd, fuse=fu)
    runs = [('frozen E1', dict(path=None, contrast=False, geometry=False)), ('V5b final', dict(path=V5B, contrast=False, geometry=False)), (final_name, F)]
    abl = 'ablation (report only, not selectable): final WITHOUT '
    alt = setup.get('report_only_alt_classifiers')
    if alt:
        for k, x in alt.items(): assert e.sha(x['path']) == x['sha256'], k + ' alt classifier changed after freezing'
        runs.append(('report only, NOT selectable: final with the other classifier set', dict(F, oval=alt['oval']['path'], marks=alt['mark']['path'], shape=alt['shape']['path'])))
    if rd != 'none': runs.append((abl + 'count reader', dict(F, reader='none')))
    if fu: runs.append((abl + 'mark fusion', dict(F, fuse=False)))
    if sh: runs.append((abl + 'shape classifier', dict(F, shape=None, reader='none')))
    if mk: runs.append((abl + 'mark classifier', dict(F, marks=None, fuse=False, classread=False)))
    for name, kw in runs:
        out, placed = run_model(rows=rows, **kw)
        e.write(OUT / ('test_rows_' + ''.join(c if c.isalnum() else '_' for c in name)[:60] + '.csv'), out)
        res[name] = dict(all=summarise(out), W02_W07=summarise([r for r in out if r['group'] == 'W02_W07']),
                         collection_v3=summarise([r for r in out if r['group'] == 'collection_v3']) if any(r['group'] == 'collection_v3' for r in out) else None, geometry_placed=placed)
    e.dump(OUT / 'test_summary.json', dict(results=res, setup=setup, run_unix=time.time(), excluded_unavailable_crops=missing))
    lines = ['| Model | Test tiers | Writers | Macro CER | Overall CER | Exact | <=1 error |', '|---|---:|---:|---:|---:|---:|---:|']
    for n, x in res.items():
        a_ = x['all']; lines.append(f"| {n} | {a_['tiers']} | {a_['writers']} | {a_['macro_writer_CER']*100:.1f}% | {a_['overall_CER']*100:.1f}% | {a_['exact']*100:.1f}% | {a_['le1']*100:.1f}% |")
    (OUT / 'test_summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8'); print('\n'.join(lines))
    for n, x in res.items(): print(n, 'per writer', {w: round(c * 100, 1) for w, c in x['all']['per_writer'].items()})


if __name__ == '__main__':
    main()
