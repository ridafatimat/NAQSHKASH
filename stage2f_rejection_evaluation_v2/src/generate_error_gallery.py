"""Failure CSV/JSON tables and development-only representative image galleries."""
import argparse
import base64
import html
import io
from integration import *
from dataset_loader import DevelopmentDataset,SPLIT,image_bytes,safe_image_path
from evaluate_rejection import load_candidate
from evaluation_metrics import grouped


def failure_groups(rows):
    def has_wrong(r):return 'DECODED_UNSUPPORTED_OR_WRONG_TIER_GLYPH' in r['raw_decision']['reason_codes']
    return dict(
        invalid_accepted=[r for r in rows if r['validity']=='INVALID' and r['actual_outcome']=='ACCEPT'],
        invalid_review=[r for r in rows if r['validity']=='INVALID' and r['actual_outcome']=='REVIEW_UNCERTAIN'],
        valid_rejected=[r for r in rows if r['validity']=='VALID' and r['actual_outcome']=='REJECT_INVALID'],
        valid_review=[r for r in rows if r['validity']=='VALID' and r['actual_outcome']=='REVIEW_UNCERTAIN'],
        correctly_rejected_invalid=[r for r in rows if r['validity']=='INVALID' and r['actual_outcome']=='REJECT_INVALID'],
        valid_accepted_wrong_transcription=[r for r in rows if r['validity']=='VALID' and r['actual_outcome']=='ACCEPT' and r['interpreted_errors'] is not None and not r['interpreted_errors']['exact']],
        D1_segmentation_failures=[r for r in rows if r['segmentation_failure']],
        preprocessing_failures=[r for r in rows if r['preprocessing_failure']],
        OCR_decoding_failures=[r for r in rows if r['ocr_failure']],
        pipeline_exceptions=[r for r in rows if r['pipeline_exception'] is not None],
        wrong_tier_OCR=[r for r in rows if has_wrong(r)],
        T1_corrections=[r for r in rows if r['corrections']],
        unscorable_or_ambiguous=[r for r in rows if r['interpretation_uncertain'] or any(e['assessment'] not in ('correct','incorrect') for e in r['correction_assessments'])])


def make_gallery(output,groups,dataset,limit):
    from PIL import Image
    chosen=[];seen=set()
    # Representative coverage across failure types, with no protected lookups.
    for group in groups.values():
        for row in group:
            if row['image_id'] not in seen:
                chosen.append(row);seen.add(row['image_id']);break
    for group in groups.values():
        for row in group:
            if row['image_id'] not in seen:chosen.append(row);seen.add(row['image_id'])
            if len(chosen)>=limit:break
        if len(chosen)>=limit:break
    page=['<!doctype html><meta charset="utf-8"><title>Rejection evaluation v2 error gallery</title><style>body{font:16px system-ui;max-width:1300px;margin:30px auto}article{border-top:1px solid #aaa;padding:20px}pre{white-space:pre-wrap}.pair{display:flex;gap:20px;flex-wrap:wrap}img{max-width:600px}details{margin:12px 0}</style>',
          '<h1>Development-only error evidence</h1><p>Actual saved decisions; no protected images. AUTO_VERIFIED labels are canonical/raster checks, not human expert review. Lists overlap. All failure records remain in CSV/JSON; display is limited.</p>']
    for row in chosen[:limit]:
        id=row['image_id']
        if id not in dataset.by_id or row['split']!=SPLIT:raise ValueError('Gallery attempted forbidden image access')
        source=dataset.by_id[id];control=dataset.by_id[source['matched_valid_control_id']]
        page.append('<article><h2>'+html.escape(id)+'</h2><p>'+html.escape(row['validity']+' / '+row['invalidity_category']+' → '+row['actual_outcome'])+'</p><div class="pair">')
        for label,entry in [('Evaluated input',source),('Matched valid clean control',control)]:
            raw=image_bytes(dataset.directory,entry)
            with Image.open(io.BytesIO(raw)) as im:
                thumb=im.convert('RGB');thumb.thumbnail((600,400));stream=io.BytesIO();thumb.save(stream,'PNG')
            href=os.path.relpath(safe_image_path(dataset.directory,entry),Path(output).parent).replace('\\','/')
            page.append('<div><p>'+label+' — <a href="'+html.escape(href,quote=True)+'">full resolution</a></p><img src="data:image/png;base64,'+base64.b64encode(stream.getvalue()).decode()+'"></div>')
        page.append('</div><pre>'+html.escape(json.dumps({k:row[k] for k in ('source_parent_id','split','canonical_rule_id','expected_outcome','actual_outcome','raw_OCR_transcription','interpreted_transcription','valid_reference_transcription','correction_assessments','decision','segmentation_result')},indent=2))+'</pre></article>')
    Path(output).write_text('\n'.join(page),encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path);p.add_argument('--split',choices=(SPLIT,),default=SPLIT)
    p.add_argument('--results-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--limit',type=int,default=24)
    a=p.parse_args()
    if a.limit<0:p.error('limit must be nonnegative')
    verify_integrity();dataset=DevelopmentDataset(a.dataset,a.split);result=load_candidate(a.results_dir,dataset)
    out=new_output(a.output_dir);groups=failure_groups(result['rows'])
    for name,rows in groups.items():write_json(out/(name+'.json'),rows);write_csv(out/(name+'.csv'),rows)
    summary,parents,strata=grouped(result['rows'])
    write_json(out/'report.json',dict(candidate=result['candidate'],summary=summary,parents=parents,strata=strata,
        failure_list_counts={k:len(v) for k,v in groups.items()},warning='Overlapping diagnostic groups; never add their counts as independent images.'))
    make_gallery(out/'error_gallery.html',groups,dataset,a.limit)
    (out/'report.md').write_text('# '+result['candidate']+' saved-results report\n\nREVIEW is distinct from REJECT. CER/exact metrics use valid-reference images only; invalid inputs use three-way semantic and structural-trigger reporting. No automatic policy tuning or selection.\n\n```json\n'+json.dumps(summary,indent=2)+'\n```\n',encoding='utf-8')
    write_json(out/'completion.json',dict(complete=True,source_results_sha256=digest(a.results_dir/'results.json'),report_sha256=digest(out/'report.json')))
    print(out)


if __name__=='__main__':main()
