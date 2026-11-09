"""Manual-only, CPU, 48-image CTC pilot; original inference is reused by hooks."""
import argparse,json,sys,uuid,html
from pathlib import Path
from collections import Counter
import numpy as np
HERE=Path(__file__).resolve().parents[1]
WORKSPACE=Path(r'C:\NAQSHKASH\stage2\stage2f_rejection_evaluation_v2')
sys.path.insert(0,str(WORKSPACE/'src'))
from integration import load_json,digest,content_hash,write_json,CHECKPOINT_SHA,make_pixel_pipeline
from dataset_loader import DevelopmentDataset,image_bytes
from observation_store import ObservationStore,exclusive_run_lock,load_raw_run
from run_ocr_baseline import deterministic_seed,execution_environment
from evaluate_r2 import verify_r2
sys.path.insert(0,str(HERE/'src'))
from alignment import path_runs,map_span,score_image,summarize,project


def check_lock():
    verify_r2()
    for p,h in load_json(HERE/'configs/experiment_lock.json').items():
        if digest(HERE/p)!=h:raise ValueError('Pilot code/config drift: '+p)
    for p,h in load_json(HERE/'configs/selection.json').get('read_only_inputs',{}).items():
        if digest(p)!=h:raise ValueError('Read-only diagnostic input drift: '+p)


def validate_selection(config,dataset):
    if config['split']!='development_design' or config['device']!='cpu':raise ValueError('Development and CPU only')
    parents=config['parents']
    if len(parents)!=12 or len({p['parent_id'] for p in parents})!=12:raise ValueError('Exactly 12 independent parents required')
    if Counter(p['category'] for p in parents)!=Counter(wrong_tier=4,invalid_count_token=4,missing_count=4):raise ValueError('Four parents per category required')
    entries=[]
    for p in parents:
        group=dataset.groups.get(p['parent_id'],[])
        if len(group)!=4 or len(p['images'])!=4 or {x['image_id'] for x in group}!={x['image_id'] for x in p['images']}:raise ValueError('Incomplete quartet')
        expected={x['image_id']:x['image_sha256'] for x in p['images']}
        for row in group:
            if row['dataset_split']!='development_design' or expected[row['image_id']]!=row['image_sha256']:raise ValueError('Selected identity/split drift')
            if row['validity_status']=='INVALID' and (row['invalidity_category']!=p['category'] or row['mutation_parameters']['row']!=p['affected_row'] or row['mutation_parameters']['column']!=p['focus_column']):raise ValueError('Affected row/category drift')
            entries.append((row,p))
    if len(entries)!=48 or config['maximum_images']!=48 or config['maximum_tier_forwards']!=144:raise ValueError('Pilot budget changed')
    if config['dataset_manifest_sha256']!=dataset.manifest_sha256 or config['checkpoint_sha256']!=CHECKPOINT_SHA:raise ValueError('Frozen input lineage changed')
    return sorted(entries,key=lambda e:e[0]['image_id'])


def load_inputs():
    check_lock();config=load_json(HERE/'configs/selection.json');dataset=DevelopmentDataset(verify_files=False)
    entries=validate_selection(config,dataset)
    raw,lineage=load_raw_run(config['baseline_raw_dir'],dataset)
    if lineage['completion']['raw_sha256']!=config['baseline_raw_sha256']:raise ValueError('Wrong baseline raw run')
    return config,dataset,entries,{x['identity']['image_id']:x['observation'] for x in raw}


def as_numpy(t):return t.detach().cpu().numpy() if hasattr(t,'detach') else np.asarray(t)


class Capture:
    """Only bytes + an offline-selected row index enter this inference adapter.

    Original D1 process_image and FrozenPipeline.observe perform all pixel work,
    resize, normalization, greedy decode and confidence computations unchanged.
    The D1 return is filtered to one row to cap model work at three tiers/image.
    """
    def __init__(self,pipeline):self.pipeline=pipeline

    def observe(self,raw,row_index):
        pipeline=self.pipeline;original=pipeline.d1.process_image;capture={};calls=[]
        def d1(image,configuration):
            out=original(image,configuration)
            capture.update(original_dimensions=list(image.shape[::-1]),processed_dimensions=list(out['image'].shape[::-1]),
                matrix=as_numpy(out['matrix']).tolist(),full_segmentation_rows=out['rows'],diagnostics=out['diagnostics'])
            chosen=out['rows'][row_index] if out['accepted'] and row_index<len(out['rows']) else []
            if len(chosen)>3:raise ValueError('More than three target tiers')
            capture['selected']=chosen;capture['pixels']=out['image']
            return dict(out,accepted=bool(out['accepted'] and chosen),rows=[chosen] if chosen else [],
                reason=out['reason'] if chosen else 'target_row_unavailable')
        def hook(module,inputs,output):
            if len(calls)>=3:raise ValueError('Tier budget exceeded')
            calls.append(dict(logits=as_numpy(output)[:,0,:].astype(np.float32),resized_dimensions=[int(inputs[0].shape[-1]),int(inputs[0].shape[-2])]))
        pipeline.d1.process_image=d1;handle=pipeline.model.register_forward_hook(hook)
        try:observation=pipeline.observe(raw)
        finally:pipeline.d1.process_image=original;handle.remove()
        if len(calls)!=len(observation.get('tiers',[])):raise ValueError('Captured forward/tier mismatch')
        capture['observation']=observation;capture['calls']=calls
        return capture


def materialize(captured,row_index,saved,directory,vocabulary):
    """Persist model evidence before any source-annotation scoring."""
    from PIL import Image
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    observation=captured['observation'];tiers=[]
    matrix=np.asarray(captured.get('matrix',np.eye(3)),dtype=float)
    for j,(call,tier) in enumerate(zip(captured['calls'],observation.get('tiers',[]))):
        logits=call['logits'];width,height=call['resized_dimensions'];box=tier['box']
        if width!=max(1,round((box[2]-box[0])*(64/float(box[3]-box[1])))):raise ValueError('Original aspect-preserving resize differs')
        if logits.shape!=(width//4,66) or height!=64 or not np.isfinite(logits).all():raise ValueError('Unexpected frozen model output shape/values')
        path=logits.argmax(-1).astype(np.int64);runs,characters=path_runs(path,vocabulary)
        prediction=''.join(s['character'] for s in characters)
        if prediction!=tier['prediction']:raise ValueError('Hook decode differs from original decoder')
        for span in characters:
            try:span['mapping']=map_span(span,box,width,matrix)
            except (ValueError,np.linalg.LinAlgError) as exc:span.update(mapping=None,mapping_error=str(exc))
        tensor_path=directory/(str(j)+'_frames.npz');np.savez_compressed(tensor_path,logits=logits,argmax_path=path)
        crop_path=directory/(str(j)+'_crop.png');x0,y0,x1,y1=box
        Image.fromarray(captured['pixels'][y0:y1,x0:x1]).save(crop_path)
        expected=saved['decoded_rows'][row_index].get(tier['role']) if row_index<len(saved['decoded_rows']) else None
        tiers.append(dict(row=row_index,role=tier['role'],box=box,crop_dimensions=[x1-x0,y1-y0],resized_dimensions=[width,height],
            padding=[0,0,0,0],frames=len(path),prediction=prediction,saved_prediction=expected,prediction_matches=prediction==expected,
            path_runs=runs,characters=characters,tensor_file=tensor_path.name,crop_file=crop_path.name,
            tensor_sha256=digest(tensor_path),crop_sha256=digest(crop_path)))
    geometry_match=captured.get('full_segmentation_rows')==saved.get('segmentation_rows')
    expected_roles={t['role'] for t in saved['segmentation_rows'][row_index]} if row_index<len(saved['segmentation_rows']) else set()
    role_match={t['role'] for t in tiers}==expected_roles
    reproduced=bool(geometry_match and role_match and tiers and all(t['prediction_matches'] for t in tiers))
    return dict(affected_row=row_index,original_dimensions=captured.get('original_dimensions'),processed_dimensions=captured.get('processed_dimensions'),
        ctc_blank_id=0,literal_space_id=next((k for k,v in vocabulary.items() if v==' '),None),logits_axes='time,class; float32; classes=66',
        source_to_processed_matrix=captured.get('matrix'),full_segmentation_rows=captured.get('full_segmentation_rows',[]),
        diagnostics=captured.get('diagnostics',{}),preprocessing_status=observation['preprocessing_status'],
        segmentation_accepted=observation['segmentation_accepted'],ocr_status=observation['ocr_status'],tiers=tiers,
        reproduction=dict(all_match=reproduced,geometry_matches=geometry_match,role_set_matches=role_match,
            mismatched_roles=[t['role'] for t in tiers if not t['prediction_matches']]),
        coordinate_convention='CV2 pixel centers first=0; source interval boundaries are pixel edges at -.5; nominal lattice, not calibrated localization')


def inside(path):
    path=Path(path).resolve()
    if not path.is_relative_to((HERE/'runs').resolve()):raise ValueError('Output must be below this pilot runs directory')
    return path


def extract(args):
    config,dataset,entries,baseline=load_inputs();out=inside(args.output_dir)
    plan=dict(schema_version=1,split='development_design',device='cpu',dataset_manifest_sha256=dataset.manifest_sha256,
        selection_sha256=content_hash([dataset.identity(r) for r,p in entries]),checkpoint_sha256=CHECKPOINT_SHA,
        config_sha256=digest(HERE/'configs/selection.json'),experiment_lock_sha256=digest(HERE/'configs/experiment_lock.json'),
        baseline_raw_sha256=config['baseline_raw_sha256'],environment=execution_environment('cpu'),
        vocabulary=load_json(WORKSPACE.parent.parent/'config/vocabulary.json'),
        vocabulary_sha256=digest(WORKSPACE.parent.parent/'config/vocabulary.json'))
    store=ObservationStore(out,plan,args.resume)
    with exclusive_run_lock(out/'run.lock'):
        pending=[e for e in entries if store.read(dataset.identity(e[0])) is None]
        adapter=Capture(make_pixel_pipeline('cpu')) if pending else None
        for n,(row,p) in enumerate(pending):
            if args.max_new_images is not None and n>=args.max_new_images:break
            deterministic_seed(row['image_id']);raw=image_bytes(dataset.directory,row)
            asset_dir=out/'assets'/uuid.uuid4().hex
            captured=adapter.observe(raw,p['affected_row'])
            observed=materialize(captured,p['affected_row'],baseline[row['image_id']],asset_dir,adapter.pipeline.vocab.index_to_char)
            observed['asset_directory']=str(asset_dir.relative_to(out))
            store.append(dataset.identity(row),observed,{})
            print('Saved',row['image_id'],'reproduced=',observed['reproduction']['all_match'],flush=True)
        ids=[dataset.identity(r) for r,p in entries];saved=sum(store.read(i) is not None for i in ids)
        if saved==48:print(json.dumps(store.finish(ids),indent=2))
        else:print('Partial:',saved,'/48; resume with identical output path and --resume')
    check_lock()


def asset_path(run,relative):
    path=(run/relative).resolve()
    if not path.is_relative_to(run.resolve()):raise ValueError('Unsafe cached asset path')
    return path


def load_extraction(run,config,dataset,entries):
    done=load_json(run/'completion.json');plan=load_json(run/'plan.json')
    if (not done['complete'] or done['records']!=48 or done['raw_sha256']!=digest(run/'observations.jsonl') or
        done['plan_sha256']!=digest(run/'plan.json') or plan['config_sha256']!=digest(HERE/'configs/selection.json') or
        plan['experiment_lock_sha256']!=digest(HERE/'configs/experiment_lock.json') or plan['checkpoint_sha256']!=CHECKPOINT_SHA or
        plan['vocabulary_sha256']!=digest(WORKSPACE.parent.parent/'config/vocabulary.json') or
        plan['dataset_manifest_sha256']!=dataset.manifest_sha256 or plan['baseline_raw_sha256']!=config['baseline_raw_sha256'] or plan['device']!='cpu' or
        plan['split']!='development_design' or plan['environment']['actual_device']!='cpu'):
        raise ValueError('Incomplete, foreign or modified extraction')
    identities=[dataset.identity(r) for r,p in entries]
    if plan['selection_sha256']!=content_hash(identities):raise ValueError('Pilot selection differs')
    store=ObservationStore(run,plan,True);items=[store.read(i) for i in identities]
    if any(i is None for i in items) or len(list((run/'records').glob('*.json')))!=48:raise ValueError('Incomplete journal')
    sealed=[json.loads(line) for line in (run/'observations.jsonl').read_text().splitlines()]
    if items!=sealed:raise ValueError('Journal differs from sealed extraction')
    if sum(len(i['observation']['tiers']) for i in items)>144:raise ValueError('Tier budget exceeded')
    for item in items:
        o=item['observation'];base=asset_path(run,o['asset_directory'])
        for t in o['tiers']:
            for key in ('tensor','crop'):
                if digest(asset_path(base,t[key+'_file']))!=t[key+'_sha256']:raise ValueError('Cached asset drift')
    return items


def gallery(cases,run,out):
    from PIL import Image,ImageDraw
    target=out/'gallery';target.mkdir();sections=[]
    for c in cases:
        if c['variant']!='clean':continue
        parts=[];capture=c['capture'];M=capture.get('source_to_processed_matrix')
        for t in capture['tiers']:
            source=asset_path(run,Path(capture['asset_directory'])/t['crop_file'])
            with Image.open(source) as im:canvas=im.convert('RGB')
            draw=ImageDraw.Draw(canvas);x0,y0,_,_=t['box']
            if M:
                for g in c['score']['reference'][t['role']]:
                    a,b,d,e=g['box'];points=project([(a-.5,b-.5),(d-.5,b-.5),(d-.5,e-.5),(a-.5,e-.5)],M)
                    draw.line([(x-x0,y-y0) for x,y in points]+[(points[0][0]-x0,points[0][1]-y0)],fill='blue',width=1)
            for s in t['characters']:
                if s['character']!=' ' and s.get('mapping'):
                    x=s['mapping']['processed_center'][0]-x0;draw.line((x,0,x,canvas.height-1),fill='red',width=1)
            name=c['image_id']+'_'+t['role']+'.png';canvas.save(target/name)
            parts.append('<p>'+html.escape(t['role']+' raw='+repr(t['prediction']))+'</p><img src="'+name+'">')
        sections.append('<details><summary>'+html.escape(c['image_id']+' '+c['validity']+' reproduced='+str(c['score']['reproduced']))+'</summary>'+''.join(parts)+'</details>')
    (target/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>CTC alignment pilot</title><style>body{font:15px system-ui}img{max-width:none}details{overflow:auto;border-top:1px solid #aaa;padding:8px}</style><h1>CTC alignment pilot — diagnostic only</h1><p>Blue: offline source ink boxes transformed into crop coordinates. Red: nominal CTC span centers; not calibrated glyph centers. Clean controls/negatives shown; illumination crops and full paths remain in extraction assets. No rejection rules.</p>'+''.join(sections),encoding='utf-8')


def evaluate(args):
    from PIL import ImageFont
    config,dataset,entries,baseline=load_inputs();run=inside(args.run_dir)
    items=load_extraction(run,config,dataset,entries);out=inside(args.output_dir);out.mkdir(parents=True,exist_ok=False)
    font_path=Path(config['font_path']);cases=[]
    for item,(record,parent) in zip(items,entries):
        if digest(font_path)!=record['layout']['font_sha256'] or record['layout']['font_size_px']!=48:raise ValueError('Offline rendering reference outside verified font scope')
        font=ImageFont.truetype(str(font_path),48);capture=item['observation']
        score=score_image(capture,record,parent['affected_row'],font,parent['focus_column'])
        cases.append(dict(image_id=record['image_id'],parent_id=parent['parent_id'],category=parent['category'],
            validity=record['validity_status'],variant='illuminated' if record['augmentation_recipe'] else 'clean',capture=capture,score=score))
    result=dict(overall=summarize(cases),by_category={k:summarize([c for c in cases if c['category']==k]) for k in sorted({c['category'] for c in cases})},
        by_variant={k:summarize([c for c in cases if c['variant']==k]) for k in ('clean','illuminated')},
        by_validity={k:summarize([c for c in cases if c['validity']==k]) for k in ('VALID','INVALID')},
        by_parent={p:summarize([c for c in cases if c['parent_id']==p]) for p in sorted({c['parent_id'] for c in cases})})
    pairs=[]
    for pid in result['by_parent']:
        for validity in ('VALID','INVALID'):
            a=next(c for c in cases if c['parent_id']==pid and c['validity']==validity and c['variant']=='clean')
            b=next(c for c in cases if c['parent_id']==pid and c['validity']==validity and c['variant']=='illuminated')
            def positions(c):return [(t['role'],s['character'],s['mapping']['source_center'][0]) for t in c['capture']['tiers'] for s in t['characters'] if s['character']!=' ' and s.get('mapping')]
            x,y=positions(a),positions(b);same=[z[:2] for z in x]==[z[:2] for z in y]
            pairs.append(dict(parent_id=pid,validity=validity,comparable=same and a['score']['reproduced'] and b['score']['reproduced'],
                source_x_shifts=[v[2]-u[2] for u,v in zip(x,y)] if same else None,
                assignment_groups_identical=(a['score']['association']['status'],a['score']['association']['groups'])==(b['score']['association']['status'],b['score']['association']['groups']),
                association_status_equal=a['score']['association']['status']==b['score']['association']['status']))
    result['paired_illumination']=pairs
    result['selection_bias']='Purposeful 12-parent development feasibility pilot; variants are not independent; long pages excluded'
    result['decision_policy']='NO_REJECTION_RULES_EXECUTED; association/order risks are offline diagnostic errors, not false reject rates'
    overall=result['overall']
    verdict=('INCONCLUSIVE_REPRODUCTION_FAILURE' if overall['reproduction_mismatches'] else
        'NOT_READY_VALID_CONTROL_ERRORS_OR_INCOMPLETE_EVIDENCE' if overall['valid_control_association_risk_images'] or overall['valid_control_order_risk_images'] or overall['valid_control_incomplete_evidence_images'] else
        'PARTIAL_FEASIBILITY_COVERAGE_OR_TOKEN_ASSIGNMENT_LIMITS' if overall['reference_glyphs_uncovered'] or overall['association_unscorable_images'] or overall['token_order_correct']<overall['token_order_denominator'] else
        'ASSOCIATION_PILOT_PROMISING_POSITION_PRECISION_REQUIRES_REVIEW')
    result['provisional_feasibility_verdict']=verdict
    write_json(out/'metrics.json',result);write_json(out/'per_image.json',cases)
    gallery(cases,run,out)
    (out/'report.md').write_text('# CTC alignment pilot — measured results\n\n'+
        'This is a development feasibility pilot, not a validated R3 rejection policy. Mismatched reproductions receive zero coverage and are excluded from conditional position/association accuracy.\n\n'+
        '~~~json\n'+json.dumps(result['overall'],indent=2)+'\n~~~\n\n'+
        'See metrics.json for category, validity, variant, parent and paired illumination results, per_image.json for ambiguous/unscorable correspondence and gallery/index.html for crop overlays.\n\n'+
        'Provisional feasibility verdict: '+verdict+'. This classifies the experiment evidence, not Talim images. No numerical localization cutoff was preregistered. Assess source-x error tails against glyph widths and association boundaries, coverage, focused mutation-token agreement and valid-control risk together. A low conditional error with poor coverage is insufficient. Zero risk flags with unscorable controls is not safety evidence. Missing-count absence is not measured by CTC blanks. Any reliable-CTC conclusion applies only to this small selected source-renderer pilot; independent image support remains necessary before definitive rejection.\n',encoding='utf-8')
    write_json(out/'completion.json',dict(complete=True,images=len(cases),metrics_sha256=digest(out/'metrics.json'),source_run_sha256=digest(run/'observations.jsonl')))
    check_lock();print(out/'report.md')


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('preflight',help='Metadata/integrity check only; no model initialization')
    p=sub.add_parser('extract');p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--resume',action='store_true');p.add_argument('--max-new-images',type=int)
    p=sub.add_parser('evaluate');p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='preflight':
        config,dataset,entries,baseline=load_inputs()
        print(json.dumps(dict(device='cpu',parents=len(config['parents']),images=len(entries),maximum_tier_forwards=144,model_initialized=False),indent=2))
    elif args.command=='extract':
        if args.max_new_images is not None and not 1<=args.max_new_images<=48:parser.error('max-new-images must be 1..48')
        extract(args)
    else:evaluate(args)


if __name__=='__main__':main()
