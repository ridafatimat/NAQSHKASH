"""Paired comparison and error gallery from actual completed T0/T1 outputs."""
import argparse
import base64
import html
from runtime import *
from measurements import grouped


def load_result(path):
    complete = load_json(path / 'completion.json')
    if not complete.get('complete') or digest(path/'results.json') != complete['results_sha256']:
        raise ValueError('Incomplete/drifted candidate')
    result = load_json(path/'results.json')
    if len(result['rows']) != complete['rows']:
        raise ValueError('Candidate count mismatch')
    return result


def compare(t0, t1):
    if t0['candidate'] != 'T0' or t1['candidate'] != 'T1':
        raise ValueError('Expected paired T0 and T1 outputs')
    for field in ('mapping_sha256', 'code_sha256'):
        if t0[field] != t1[field]: raise ValueError('Candidate experiment lineage differs: ' + field)
    for field in ('manifest_sha256', 'observations_sha256', 'checkpoint_sha256'):
        if t0['baseline_lineage'][field] != t1['baseline_lineage'][field]:
            raise ValueError('Unpaired input lineage: ' + field)
    a, b = {r['image_id']:r for r in t0['rows']}, {r['image_id']:r for r in t1['rows']}
    if set(a) != set(b) or len(a) != len(t0['rows']) or len(b) != len(t1['rows']):
        raise ValueError('Coverage differs or contains duplicate IDs')
    for key in a:
        if (a[key]['raw_ocr_rows'], a[key]['evaluation_truth_rows'], a[key]['source_parent_id']) != (
                b[key]['raw_ocr_rows'], b[key]['evaluation_truth_rows'], b[key]['source_parent_id']):
            raise ValueError('Raw prediction, source parent or offline truth differs')
    summary0, parents0, _ = grouped(t0['rows'])
    summary1, parents1, _ = grouped(t1['rows'])
    changed = [b[k] for k in b if b[k]['corrections'] or a[k]['decision'] != b[k]['decision']]
    return dict(T0=summary0, T1=summary1, changed_images=len(changed),
                parent_results=dict(T0=parents0,T1=parents1), independent_test=False), changed


def gallery(path, changed, limit):
    rows = ['<!doctype html><meta charset="utf-8"><title>T0/T1 saved-output comparison</title><style>body{font:16px system-ui;max-width:1100px;margin:30px auto}article{border-top:1px solid #aaa;padding:20px}pre{white-space:pre-wrap}img{max-width:650px;max-height:500px}</style>',
            '<h1>Contextual edits and decision changes</h1><p>Exposed development evidence. ACCEPT is not proof of exact transcription. All changes remain in comparison.json; this gallery is display-limited.</p>']
    for item in changed[:limit]:
        rows.append('<article><h2>' + html.escape(item['image_id']) + '</h2>')
        if item.get('original_image_path'):
            raw = asset_bytes(item['original_image_path'])
            mime = 'image/jpeg' if raw[:2]==b'\xff\xd8' else 'image/png'
            rows.append('<img src="data:' + mime + ';base64,' + base64.b64encode(raw).decode() + '">')
        detail = {k:item[k] for k in ('raw_ocr_rows','interpreted_ocr_rows','evaluation_truth_rows','correction_assessments','raw_errors','interpreted_errors','raw_f1_decision','decision')}
        rows.append('<pre>' + html.escape(json.dumps(detail, indent=2)) + '</pre></article>')
    path.write_text('\n'.join(rows),encoding='utf-8')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--t0-dir',type=Path,required=True)
    ap.add_argument('--t1-dir',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--gallery-limit',type=int,default=24)
    args = ap.parse_args()
    if args.gallery_limit < 0: ap.error('gallery-limit must be nonnegative')
    verify_integrity()
    t0,t1 = load_result(args.t0_dir),load_result(args.t1_dir)
    if t0['baseline_lineage']['manifest_sha256'] != digest(LEGACY/'manifests/development_manifest.csv'):
        raise ValueError('Only frozen development comparisons are permitted')
    report,changed = compare(t0,t1)
    out = new_output(args.output_dir)
    write_json(out/'comparison.json',dict(summary=report,changed_images=changed))
    serialize_csv(out/'summary.csv',[dict(candidate=k,**report[k]) for k in ('T0','T1')])
    serialize_csv(out/'parent_comparison.csv',[dict(candidate=k,source_parent_id=p,**v) for k in ('T0','T1') for p,v in report['parent_results'][k].items()])
    gallery(out/'error_gallery.html',changed,args.gallery_limit)
    lines=['# T0 versus T1 — saved development outputs','',
           'Review-to-accept transitions alone are not improvements. Inspect exact accuracy, CER, accepted correctness, harmful edits and parent/cohort results.','']
    for name in ('T0','T1'):
        lines += ['## '+name,'','```json',json.dumps(report[name],indent=2),'```','']
    (out/'comparison.md').write_text('\n'.join(lines),encoding='utf-8')
    write_json(out/'completion.json',dict(complete=True,comparison_sha256=digest(out/'comparison.json')))
    print(out)


if __name__=='__main__':main()
