"""Development-only saved-evidence R3 evaluation, optional overlays, freeze preparation."""
import argparse,copy,json,platform,sys,uuid
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter,defaultdict
from importlib.metadata import version
from policy_v3 import HERE,WORKSPACE,STAGE1,ID,load_policy,verify_dependencies,evidence,decide,assert_invariants
from integration import load_json,digest,write_json,write_csv,content_hash
from evaluate_r2 import verify_r2
sys.path.insert(0,str(STAGE1/'src'))
from evaluate_text_alignment import load_inputs as load_R2
from evaluation_metrics import grouped
from review_overlay import geometry_index,plan_regions,generate_gallery
from localization_metrics import score as score_location,summarize as summarize_locations


def safe_output(path):
    p=Path(path).resolve()
    if not p.is_relative_to((HERE/'runs').resolve()):raise ValueError('New output must stay within r3_hitl_v3/runs')
    return p


def load_sources(config,r2_dir=None,stage1_dir=None):
    verify_dependencies(config);dataset,rows,raw,provenance=load_R2(r2_dir)
    if stage1_dir is None:
        candidates=[p for p in (STAGE1/'runs').iterdir() if p.is_dir() and (p/'completion.json').is_file() and load_json(p/'completion.json').get('complete')]
        if not candidates:raise ValueError('No completed Stage 1 run')
        stage1_dir=max(candidates,key=lambda p:((p/'completion.json').stat().st_mtime_ns,p.name))
    stage1_dir=Path(stage1_dir).resolve()
    if not stage1_dir.is_relative_to((STAGE1/'runs').resolve()):raise ValueError('Use completed development Stage 1 evidence')
    done=load_json(stage1_dir/'completion.json');metrics=load_json(stage1_dir/'metrics.json');p=metrics['provenance']
    if (not done['complete'] or done['images']!=2800 or done['metrics_sha256']!=digest(stage1_dir/'metrics.json') or done['evidence_sha256']!=digest(stage1_dir/'per_image.jsonl') or
        p['r2_results_sha256']!=provenance['r2_results_sha256'] or p['raw_sha256']!=provenance['raw_sha256'] or
        p['code_sha256']['text_alignment.py']!=digest(STAGE1/'src/text_alignment.py') or
        p['code_sha256']['evaluate_text_alignment.py']!=digest(STAGE1/'src/evaluate_text_alignment.py') or
        p['max_offset']!=config['max_offset'] or p['max_candidates']!=config['max_candidates']):raise ValueError('Stage 1 evidence/dependency/budget lineage mismatch')
    saved=[json.loads(line) for line in (stage1_dir/'per_image.jsonl').read_text().splitlines()]
    if len(saved)!=2800 or {r['image_id'] for r in saved}!=set(dataset.by_id):raise ValueError('Stage 1 development coverage differs')
    baseline={r['image_id']:r for r in rows}
    for r in saved:
        if r['original_R2_outcome']!=baseline[r['image_id']]['actual_outcome']:raise ValueError('Stage 1/R2 outcome mismatch')
    provenance.update(stage1_dir=str(stage1_dir),stage1_metrics_sha256=done['metrics_sha256'],stage1_evidence_sha256=done['evidence_sha256'])
    return dataset,rows,raw,{r['image_id']:r for r in saved},provenance


def process_case(base,observation,saved,source,config,geometry=None):
    # Decision and finding generation precede both GT scoring and pixel access.
    ev=evidence(base['interpreted_transcription'],observation,config,verified_saved_analysis=saved['analysis'])
    result=decide(base['decision'],base['interpreted_transcription'],observation,config,ev)
    regions=plan_regions(result['findings'],geometry)
    r=dict(image_id=base['image_id'],source_parent_id=base['source_parent_id'],image_sha256=base['image_sha256'],split='development_design',
        R2=copy.deepcopy(base['decision']),Stage1_outcome=saved['review_only_counterfactual'],R3=result['decision'],policy_id=ID,
        findings=result['findings'],alignment_evidence=result['alignment_evidence'],new_review=result['new_review'],review_regions=regions,
        interpreted_transcription=base['interpreted_transcription'],OCR_available=observation['preprocessing_status']=='ok' and observation['segmentation_accepted'] and observation['ocr_status']=='ok')
    # All following fields are evaluation-only, never sent back to policy/regions.
    r.update(validity=source['validity_status'],category=source['invalidity_category'],variant='illuminated' if source['augmentation_recipe'] else 'clean',
        valid_transcription_problem=source['validity_status']=='VALID' and bool(base.get('interpreted_errors') and not base['interpreted_errors']['exact']))
    r['localization']=score_location(r,source)
    return r


def metrics(records,bases):
    byid={r['image_id']:r for r in records};versions={}
    for candidate in ('R2','Stage1','R3_HITL_V3'):
        rows=[]
        for b in bases:
            r=copy.deepcopy(b);result=byid[b['image_id']]
            r['actual_outcome']=result['R2']['outcome'] if candidate=='R2' else result['Stage1_outcome'] if candidate=='Stage1' else result['R3']['outcome']
            if candidate=='R3_HITL_V3':r['triggered_reason_codes']=result['R3']['reason_codes']
            rows.append(r)
        total,parents,groups=grouped(rows)
        total['outcome_counts_and_rates']={v:{o:dict(count=total['three_way_confusion'][v][o],rate=total['three_way_confusion'][v][o]/sum(total['three_way_confusion'][v].values())) for o in ('ACCEPT','REVIEW_UNCERTAIN','REJECT_INVALID')} for v in ('VALID','INVALID') if sum(total['three_way_confusion'][v].values())}
        versions[candidate]=dict(summary=total,parents=parents,groups=groups)
    transitions=Counter((r['R2']['outcome'],r['R3']['outcome']) for r in records)
    changed=[]
    for r in records:
        assert_invariants(r['R2']['outcome'],r['R3']['outcome'])
        if r['Stage1_outcome']!=r['R3']['outcome']:
            changed.append(dict(image_id=r['image_id'],source_parent_id=r['source_parent_id'],validity=r['validity'],category=r['category'],
                Stage1=r['Stage1_outcome'],R3=r['R3']['outcome'],reasons=[f['reason_code'] for f in r['findings']],
                explanation='R3 accepts a supported complete canonical witness; incomplete or unsupported evidence remains fail-safe REVIEW even when Stage 1 preserved an unknown case.',
                page_candidates=r['alignment_evidence']['analysis']['page_candidate_count_in_search']))
    reasons=Counter(f['reason_code'] for r in records for f in r['findings'])
    result=dict(candidates=versions,R2_to_R3_transitions=[dict(before=a,after=b,images=n) for (a,b),n in sorted(transitions.items())],
        R2_reviews_preserved=sum(r['R2']['outcome']=='REVIEW_UNCERTAIN' and r['R3']['outcome']=='REVIEW_UNCERTAIN' for r in records),
        new_R3_reviews=sum(r['new_review'] for r in records),new_definitive_rejections=sum(r['R2']['outcome']!='REJECT_INVALID' and r['R3']['outcome']=='REJECT_INVALID' for r in records),
        policy_invariants_pass=True,reason_counts=dict(reasons),Stage1_differences=changed,
        all_REVIEW_reason_counts=dict(Counter(code for r in records if r['R3']['outcome']=='REVIEW_UNCERTAIN' for code in set(r['R3'].get('reason_codes',[])))),
        missing_alignment_evidence=0,
        unavailable_alignment_reasons=dict(Counter(r['alignment_evidence']['analysis']['unsupported_reason'] for r in records if r['alignment_evidence']['analysis'].get('unsupported_reason'))),
        complete_text_searches=sum(r['alignment_evidence']['analysis']['all_rows_exhaustive_in_literal_model'] for r in records),
        incomplete_or_unsupported=sum(not r['alignment_evidence']['analysis']['all_rows_exhaustive_in_literal_model'] for r in records),
        unsupported_reasons=dict(Counter(row.get('unsupported_reason','UNKNOWN') for r in records for row in r['alignment_evidence']['analysis']['rows'] if row['status']=='unsupported')),
        unavailable_OCR=sum(not r['OCR_available'] for r in records),
        valid_images_with_transcription_problems=sum(r['valid_transcription_problem'] for r in records),
        valid_reviews_with_transcription_problems=sum(r['valid_transcription_problem'] and r['R3']['outcome']=='REVIEW_UNCERTAIN' for r in records))
    result['by_OCR_availability']={str(v):dict(images=sum(r['OCR_available']==v for r in records),outcomes=dict(Counter(r['R3']['outcome'] for r in records if r['OCR_available']==v))) for v in (True,False)}
    result['localization']=summarize_locations(records)
    result['localization_by_parent']={p:summarize_locations([r for r in records if r['source_parent_id']==p]) for p in sorted({r['source_parent_id'] for r in records})}
    return result


def fingerprint(config):
    files=[*sorted((HERE/'src').glob('*.py')),*sorted((HERE/'configs').glob('*.json')),*sorted((HERE/'tests').glob('*.py')),HERE/'README.md',HERE/'reports/evaluation_protocol.md']
    if (HERE/'reports/test_results.json').exists():files.append(HERE/'reports/test_results.json')
    return {str(p):digest(p) for p in files if p.is_file()}


def verify_completed(run):
    run=safe_output(run);done=load_json(run/'completion.json')
    if not done.get('complete'):raise ValueError('Run incomplete')
    for name,h in done['outputs'].items():
        p=(run/name).resolve()
        if not p.is_relative_to(run) or digest(p)!=h:raise ValueError('Run output changed: '+name)
    manifest=load_json(run/'run_manifest.json')
    if manifest['split']!='development_design' or manifest['policy_id']!=ID:raise ValueError('Not a development R3 run')
    return run,manifest,done


def evaluate(args):
    config=load_policy();dataset,bases,raw,stage1,source=load_sources(config,args.r2_dir,args.stage1_dir)
    if args.preflight:
        print(json.dumps(dict(policy=ID,images=len(bases),parents=len(dataset.groups),split='development_design',source=source,complete_evaluation_executed=False),indent=2));return
    out=safe_output(args.output_dir);out.mkdir(parents=True,exist_ok=False)
    initial=fingerprint(config)
    manifest=dict(policy_id=ID,split='development_design',run_identifier=out.name,created_UTC=datetime.now(timezone.utc).isoformat(),
        initial_status='IN_PROGRESS',completion_status_file='completion.json',images=len(bases),parents=len(dataset.groups),config=config,source=source,source_code_and_protocol_sha256=initial,
        dependency_sha256=config['dependencies'],software=dict(python=platform.python_version(),numpy=version('numpy'),Pillow=version('Pillow')),
        policy_selection='supported complete canonical witness; uniqueness not required; incomplete evidence remains REVIEW; no confidence threshold')
    write_json(out/'run_manifest.json',manifest)
    geometry,geometry_log=geometry_index(dataset,raw,source['raw_sha256']);write_json(out/'geometry_provenance.json',geometry_log)
    records=[process_case(b,raw[b['image_id']],stage1[b['image_id']],dataset.by_id[b['image_id']],config,geometry.get(b['image_id'])) for b in bases]
    measured=metrics(records,bases)
    write_json(out/'metrics.json',measured)
    with (out/'per_image.jsonl').open('x',encoding='utf-8') as stream:
        for r in records:stream.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
    write_csv(out/'Stage1_differences.csv',measured['Stage1_differences'])
    write_csv(out/'outcomes.csv',[dict(image_id=r['image_id'],parent=r['source_parent_id'],validity=r['validity'],category=r['category'],variant=r['variant'],
        R2=r['R2']['outcome'],Stage1=r['Stage1_outcome'],R3=r['R3']['outcome'],new_review=r['new_review'],reason_codes=r['R3']['reason_codes'],region_available=bool(r['review_regions']['regions'])) for r in records])
    summaries={k:v['summary'] for k,v in measured['candidates'].items()}
    (out/'report.md').write_text('# R3_HITL_V3 development results\n\nREVIEW is not definitive rejection. This policy permits any adequately supported complete canonical page interpretation. Uniqueness is not required; incomplete or unsupported evidence remains REVIEW. Differences from Stage 1 are listed without forcing target counts. Image validity is not proved by text consistency.\n\n~~~json\n'+json.dumps(summaries,indent=2)+'\n~~~\n\n'+
        'See metrics.json for transitions, category/parent/variant/transcription/coverage/localization metrics; Stage1_differences.csv identifies every changed image. Precise pixel localization has no scoring coverage; broad row/tier overlap is a separate metric. Optional overlays require verified source transforms and may be unavailable. No validation/protected run or freeze was performed.\n',encoding='utf-8')
    verify_r2();verify_dependencies(config)
    for p,h in initial.items():
        if digest(p)!=h:raise ValueError('R3 implementation changed during evaluation')
    for p,h in [(Path(source['r2_dir'])/'results.json',source['r2_results_sha256']),(Path(source['stage1_dir'])/'per_image.jsonl',source['stage1_evidence_sha256'])]:
        if digest(p)!=h:raise ValueError('Historical source changed during evaluation')
    names=('run_manifest.json','geometry_provenance.json','metrics.json','per_image.jsonl','Stage1_differences.csv','outcomes.csv','report.md')
    write_json(out/'completion.json',dict(complete=True,status='COMPLETE',policy_id=ID,images=len(records),parents=len(dataset.groups),outputs={n:digest(out/n) for n in names}))
    print(out/'report.md')


def overlays(args):
    from dataset_loader import DevelopmentDataset
    run,manifest,done=verify_completed(args.run_dir);dataset=DevelopmentDataset(verify_files=False)
    if manifest['source']['dataset_manifest_sha256']!=dataset.manifest_sha256:raise ValueError('Overlay dataset differs')
    records=[json.loads(x) for x in (run/'per_image.jsonl').read_text().splitlines()]
    if len(records)!=2800 or {r['image_id'] for r in records}!=set(dataset.by_id):raise ValueError('Not exact development coverage')
    target=run/'overlays'/('gallery_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:8])
    summary=generate_gallery(records,dataset,target,args.limit,args.image_id)
    write_json(target/'overlay_manifest.json',dict(source_completed_run=str(run),decision_evidence_sha256=digest(run/'per_image.jsonl'),**summary))
    print(target/'index.html')


def prepare_freeze(args):
    if not args.ack_development_reviewed:raise ValueError('Explicit --ack-development-reviewed is required; no automatic freeze')
    run,manifest,done=verify_completed(args.run_dir);measured=load_json(run/'metrics.json');config=load_policy();verify_dependencies(config);verify_r2()
    if not measured['policy_invariants_pass'] or measured['new_definitive_rejections']!=0:raise ValueError('Development invariants failed')
    proof=load_json(HERE/'reports/test_results.json')
    tested={p.relative_to(HERE).as_posix():digest(p) for folder in ('src','configs','tests') for p in sorted((HERE/folder).glob('*')) if p.is_file() and p.suffix in ('.py','.json')}
    if proof.get('failed')!=0 or proof.get('passed',0)<1 or proof.get('source_sha256')!=tested:raise ValueError('Current implementation lacks matching passed-test proof')
    current=fingerprint(config)
    if current!=manifest['source_code_and_protocol_sha256'] or config!=manifest['config']:raise ValueError('Code/protocol changed after reviewed development run')
    folder=HERE/'runs/freeze_candidates';folder.mkdir(parents=True,exist_ok=True)
    target=folder/('manifest_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:8]+'.json')
    write_json(target,dict(status='CANDIDATE_MANIFEST_NOT_AN_APPROVED_FREEZE',policy_id=ID,created_UTC=datetime.now(timezone.utc).isoformat(),
        operator_acknowledged_development_review=True,reviewed_run=str(run),reviewed_run_completion_sha256=digest(run/'completion.json'),
        source_config_protocol_tests_sha256=current,dependencies_sha256=config['dependencies'],
        report_templates_sha256={p.name:digest(p) for p in sorted((HERE/'reports').glob('*template.md'))},
        next_step='Separate reviewer approval record referencing this manifest SHA; no held-out evaluation authorized by this command'))
    print(target)


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('evaluate');p.add_argument('--split',choices=('development_design',),default='development_design');p.add_argument('--r2-dir',type=Path);p.add_argument('--stage1-dir',type=Path);p.add_argument('--preflight',action='store_true');p.add_argument('--output-dir',type=Path)
    p=sub.add_parser('overlays');p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--limit',type=int,default=6);p.add_argument('--image-id',action='append')
    p=sub.add_parser('prepare-freeze');p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--ack-development-reviewed',action='store_true')
    args=parser.parse_args()
    if args.command=='evaluate':
        if not args.preflight and args.output_dir is None:parser.error('Evaluation requires --output-dir')
        evaluate(args)
    elif args.command=='overlays':overlays(args)
    else:prepare_freeze(args)


if __name__=='__main__':main()
