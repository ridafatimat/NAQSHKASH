"""Matched examples, raw structures and captions outside the model images."""
import argparse
import html
import os
import base64
import io
from common_v2 import *


def create_gallery(dataset,output,limit=40):
    dataset=Path(dataset);output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    rows=read_json(dataset/'manifest.json')
    by={r['image_id']:r for r in rows}
    invalid=[r for r in rows if r['augmentation_recipe'] is None and r['validity_status']=='INVALID' and r['dataset_split']!='protected_test']
    page=['<!doctype html><meta charset="utf-8"><title>Invalid Talim v2 matched review</title><style>body{font:16px system-ui;max-width:1400px;margin:30px auto;color:#17222c}.pair{display:grid;grid-template-columns:1fr 1fr;gap:18px}img{max-width:100%;border:1px solid #ccc}article{border-top:2px solid #899;padding:24px 0}pre{white-space:pre-wrap;background:#eef2f5;padding:12px}summary{cursor:pointer}</style>',
          '<h1>Dataset v2 — matched visual review</h1><p>AUTO_VERIFIED means fixed-font canonical-rule and raster verification, not human/domain-expert approval. No captions appear inside model images. Protected test examples are never displayed.</p>']
    for r in invalid[:limit]:
        valid=by[r['matched_valid_control_id']]
        vp=os.path.relpath(dataset/valid['image_file_path'],output.parent).replace('\\','/')
        ip=os.path.relpath(dataset/r['image_file_path'],output.parent).replace('\\','/')
        page += ['<article><h2>'+html.escape(r['invalidity_category']+' — '+r['source_parent_id'])+'</h2>',
                 '<p>'+html.escape(json.dumps(dict(rule=r['canonical_rule_id'],location=r['mutation_parameters'],status=r['verification_status'],split=r['dataset_split'])))+'</p>',
                 '<div class="pair"><div><h3>VALID matched control</h3><img src="'+vp+'"></div><div><h3>INVALID visible notation</h3><img src="'+ip+'"></div></div>',
                 '<p>'+html.escape(json.dumps(r['verification_evidence']))+'</p>']
        # Inline full-resolution mutation zooms keep dense/long examples readable
        # without writing additional OCR dataset images or answer text into them.
        layout=r['layout'];params=r['mutation_parameters']
        x=max(0,layout['padding']+layout['extra_left']+(params['column']-1)*layout['cell_width'])
        y=layout['padding']+params['row']*(3*layout['line_height']+layout['row_gap'])
        box=(x,max(0,y-5),min(layout['width'],x+(params['width']+3)*layout['cell_width']),min(layout['height'],y+3*layout['line_height']+5))
        page.append('<p>Full-resolution mutation-region zooms (valid, then invalid). Click the full-image links for long sequences: <a href="'+vp+'">valid PNG</a> / <a href="'+ip+'">invalid PNG</a>.</p><div class="pair">')
        for entry in (valid,r):
            with Image.open(dataset/entry['image_file_path']) as im:
                stream=io.BytesIO();im.crop(box).save(stream,'PNG')
            page.append('<img src="data:image/png;base64,'+base64.b64encode(stream.getvalue()).decode()+'">')
        page.append('</div>')
        scans=[s for s in rows if s['source_parent_id']==r['source_parent_id'] and s['augmentation_recipe']]
        for scan in scans:
            url=os.path.relpath(dataset/scan['image_file_path'],output.parent).replace('\\','/')
            page += ['<details><summary>'+html.escape(scan['validity_status']+' illumination variant — '+scan['verification_status'])+'</summary><img src="'+url+'"></details>']
        page += ['<details><summary>Original and mutated structured notation</summary><pre>'+html.escape(json.dumps(dict(original=r['original_structured_notation'],mutated=r['mutated_structured_notation']),indent=2))+'</pre></details></article>']
    page.append('<p>Review checklist: verify actual tier/count change; readable glyphs; compare source/control; reject alias-only defects; confirm grouping under the stated canonical scope; mark domain disagreement; record reviewer identity/date separately. No human review has yet occurred.</p>')
    with output.open('x',encoding='utf-8') as f:f.write('\n'.join(page))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--limit',type=int,default=40)
    a=p.parse_args()
    if not a.output.resolve().is_relative_to(ROOT.resolve()):raise ValueError('Gallery must stay inside v2 workspace')
    create_gallery(a.dataset,a.output,a.limit)


if __name__=='__main__':main()
