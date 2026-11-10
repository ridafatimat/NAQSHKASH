"""Frozen R3 held-out adapter. No command defaults to a split or authorization."""
import argparse,copy,json,sys,uuid,platform
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
from release_io import HERE,RELEASE,verify,load,sha,canonical,write_json,output_path,check_completion
from heldout_dataset import HeldoutDataset,SPLITS,identity,rendering_metadata
from journal import Journal,read_completed

TOKENS={'validation_calibration':'I_AUTHORIZE_VALIDATION','protected_test':'I_AUTHORIZE_PROTECTED_TEST_AFTER_VALIDATION_REVIEW'}


def authorize(args):
    if args.authorization!=TOKENS[args.split]:raise ValueError('Explicit matching --authorization is required; no held-out action occurred')
    if args.split=='protected_test':
        if not args.ack_validation_reviewed or args.validation_run is None:raise ValueError('Protected test requires a completed --validation-run and --ack-validation-reviewed')
        done=check_completion(args.validation_run,'evaluation')
        if done['split']!='validation_calibration' or done['records']!=600:raise ValueError('Not the frozen 600-image validation result')
        marker=load(HERE/'state/validation_calibration/evaluation_completed.json')
        if marker['completion_sha256']!=sha(Path(args.validation_run)/'completion.json'):raise ValueError('Validation result is not the one-time registered result')


def frozen_runtime():
    from freeze_runtime import FrozenRuntime
    return FrozenRuntime(RELEASE)


def environment(runtime):
    import importlib,scipy
    helper=importlib.import_module('run_ocr_baseline')
    actual=helper.execution_environment('cpu');expected=runtime.manifest['inference_environment']
    if actual!=expected:raise ValueError('Inference software/device differs from approved environment: '+repr(actual))
    if scipy.__version__!=runtime.manifest['additional_software']['scipy']:raise ValueError('SciPy version differs from freeze')
    return actual


def observe(pipeline,raw):
    """Original observe(bytes), with a read-only D1-return geometry hook."""
    import numpy as np
    original=pipeline.d1.process_image;geometry={}
    def capture(image,configuration):
        result=original(image,configuration)
        M=np.asarray(result['matrix'],dtype=float)
        if M.shape==(3,3) and np.isfinite(M).all() and abs(np.linalg.det(M))>1e-12:
            geometry.update(verified=True,source_dimensions=list(image.shape[::-1]),processed_dimensions=list(result['image'].shape[::-1]),
                            source_to_processed_matrix=M.tolist(),rows=copy.deepcopy(result['rows']),provenance='unchanged frozen D1 process_image return; no annotation geometry')
        return result
    pipeline.d1.process_image=capture
    try:observation=pipeline.observe(raw)
    finally:pipeline.d1.process_image=original
    if geometry.get('rows')!=observation.get('segmentation_rows'):geometry={}
    return observation,geometry or None


def claim_path(split):return HERE/'state'/split/'inference_claim.json'


def claim_inference(args,plan):
    claim=claim_path(args.split);out=output_path(args.output_dir)
    if (claim.parent/'evaluation_completed.json').exists():raise ValueError('Split evaluation already complete; no repeat inference permitted')
    binding=dict(split=args.split,raw_directory=str(out),release_sha256=plan['release_sha256'],selection_sha256=plan['selection_sha256'],plan_sha256=canonical(plan))
    if claim.exists():
        if not args.resume or load(claim)!=binding:raise ValueError('Split already claimed; resume only its identical output and frozen plan')
    else:
        if args.resume or out.exists():raise ValueError('New split inference requires a new output directory; no overwrite')
        write_json(claim,binding)
    return out


def preflight(args):
    m=verify();dataset=HeldoutDataset(m,args.split);runtime=frozen_runtime()
    sources=runtime.inspect_inference_imports();env=environment(runtime)
    print(json.dumps(dict(preflight_passed=True,split=args.split,selection_images=len(dataset.rows),selection_parents=len(dataset.groups),
                          split_disjointness=dataset.audit,release_sha256=sha(RELEASE/'release_manifest.json'),checkpoint_sha256=m['checkpoint']['sha256'],
                          inference_environment=env,heldout_observations_at_release=m['heldout_observation_inventory']['compatible_heldout_observations'],
                          manual_inference_required=True,image_bytes_read=0,model_initialized=False,OCR_inference=False,
                          frozen_local_imports=sources),indent=2))


def infer(args):
    authorize(args);m=verify();dataset=HeldoutDataset(m,args.split);runtime=frozen_runtime();env=environment(runtime)
    release_sha=sha(RELEASE/'release_manifest.json')
    plan=dict(schema_version=1,kind='authorized_heldout_raw',split=args.split,dataset_manifest_sha256=dataset.manifest_sha256,selection_sha256=dataset.selection_sha256,
              release_sha256=release_sha,policy_id='R3_HITL_V3',checkpoint_sha256=m['checkpoint']['sha256'],execution_environment=env,
              authorization=args.authorization,validation_reviewed_completion_sha256=sha(Path(args.validation_run)/'completion.json') if args.split=='protected_test' else None,
              inference_contract='original FrozenPipeline.observe(bytes), CPU, unchanged D1/E1, same per-record seed; no labels/targets',records=600,parents=150)
    from observation_store import exclusive_run_lock
    import importlib
    helper=importlib.import_module('run_ocr_baseline')
    lock=HERE/'state'/args.split/'execution.lock';lock.parent.mkdir(parents=True,exist_ok=True)
    with exclusive_run_lock(lock):
        out=claim_inference(args,plan);store=Journal(out,plan,args.resume)
        pending=[r for r in dataset.rows if store.read(identity(r)) is None]
        if (out/'completion.json').exists():
            check_completion(out,'raw');print('Raw inference is already complete; no model loaded.');return
        pipeline=runtime.pipeline() if pending else None
        for index,row in enumerate(pending,1):
            raw=dataset.image_bytes(row['image_id']);helper.deterministic_seed(row['image_id'])
            try:observation,geometry=observe(pipeline,raw)
            except (ValueError,RuntimeError,OSError) as exc:observation=runtime.integration.pipeline_exception(exc);geometry=None
            context=runtime.rendering_context(rendering_metadata(row))
            store.append(identity(row),observation,context,geometry)
            if index%25==0:print('Saved new records',index,'/',len(pending),flush=True)
        verify();done=store.finish(dataset)
        print(json.dumps(dict(raw_directory=str(out),**done),indent=2))


def process_record(runtime,item,row,paired_category):
    # Decisions and image-region planning occur BEFORE offline source scoring.
    observation=item['observation'];decision=runtime.decide(observation,item['rendering_context'])
    r3=decision['R3'];regions=runtime.regions.plan_regions(r3['findings'],item.get('geometry'))
    record=dict(image_id=item['identity']['image_id'],source_parent_id=item['identity']['source_parent_id'],image_sha256=item['identity']['image_sha256'],
                split=item['identity']['dataset_split'],R2=decision['R2'],R3=r3['decision'],findings=r3['findings'],alignment_evidence=r3['alignment_evidence'],
                interpreted_transcription=decision['text'],new_review=r3['new_review'],review_regions=regions,
                OCR_available=observation.get('preprocessing_status')=='ok' and observation.get('segmentation_accepted') is True and observation.get('ocr_status')=='ok',
                carpet_generation_ready=False,selected_carpet_reading=None,ambiguous_text_alignment=decision['ambiguous_text_alignment'],
                carpet_generation_block_reason=decision['carpet_generation_block_reason'])
    # Existing evaluation function now receives source data solely for scoring.
    import importlib
    base=importlib.import_module('evaluate_rejection').evaluate_case(item,row,'R1',paired_category)
    if base['decision']!=decision['R1']['final_decision'] or base['interpreted_transcription']!=decision['text']:raise AssertionError('Offline data altered runtime interpretation')
    record.update(validity=row['validity_status'],category=row['invalidity_category'],variant=base['variant'])
    record['localization']=runtime.localization.score(record,row)
    base.update(actual_outcome=record['R3']['outcome'],decision=record['R3'],triggered_reason_codes=record['R3']['reason_codes'])
    return record,base


def evaluate(args):
    authorize(args);m=verify();dataset=HeldoutDataset(m,args.split);release_sha=sha(RELEASE/'release_manifest.json')
    claim=load(claim_path(args.split))
    if Path(claim['raw_directory']).resolve()!=Path(args.raw_dir).resolve() or claim['release_sha256']!=release_sha:raise ValueError('Raw directory is not the authorized split claim')
    items,raw_plan=read_completed(args.raw_dir,dataset,release_sha);runtime=frozen_runtime();environment(runtime)
    from observation_store import exclusive_run_lock
    import importlib
    state=HERE/'state'/args.split;lock=state/'execution.lock'
    with exclusive_run_lock(lock):
        if (state/'evaluation_completed.json').exists():raise ValueError('Split already evaluated; completed results cannot be overwritten or rerun')
        out=output_path(args.output_dir)
        if (state/'evaluation_claim.json').exists():raise ValueError('A scoring attempt is already registered; preserve it and stop rather than repeat held-out scoring')
        if out.exists():raise FileExistsError('Evaluation output already exists; no overwrite')
        write_json(state/'evaluation_claim.json',dict(output_directory=str(out),raw_completion_sha256=sha(Path(args.raw_dir)/'completion.json'),release_sha256=release_sha,split=args.split))
        out.mkdir(parents=True,exist_ok=False)
        manifest=dict(policy_id='R3_HITL_V3',kind='evaluation',split=args.split,run_identifier=out.name,created_UTC=datetime.now(timezone.utc).isoformat(),
                      status='IN_PROGRESS',completion_status_file='completion.json',release_sha256=release_sha,raw_directory=str(Path(args.raw_dir).resolve()),
                      raw_completion_sha256=sha(Path(args.raw_dir)/'completion.json'),raw_plan=raw_plan,dataset_manifest_sha256=dataset.manifest_sha256,
                      selection_sha256=dataset.selection_sha256,source_config_dependency_hashes=m['files'],checkpoint=m['checkpoint'],records=600,parents=150,
                      authorization=args.authorization,carpet_contract='R3 ACCEPT is not physical alignment or carpet-generation authorization')
        write_json(out/'run_manifest.json',manifest)
        categories={p:next(r['invalidity_category'] for r in group if r['validity_status']=='INVALID') for p,group in dataset.groups.items()}
        records=[];bases=[]
        with (out/'per_image.jsonl').open('x',encoding='utf-8') as stream:
            for item in items:
                row=dataset.by_id[item['identity']['image_id']];record,base=process_record(runtime,item,row,categories[row['source_parent_id']])
                records.append(record);bases.append(base);stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
        metric_module=importlib.import_module('evaluation_metrics');summary,parents,groups=metric_module.grouped(bases)
        summary['outcome_counts_and_rates']={v:{o:dict(count=n,rate=n/sum(counts.values())) for o,n in counts.items()} for v,counts in summary['three_way_confusion'].items()}
        transitions=Counter((r['R2']['outcome'],r['R3']['outcome']) for r in records)
        measured=dict(summary=summary,parents=parents,groups=groups,R2_to_R3_transitions=[dict(before=a,after=b,images=n) for (a,b),n in sorted(transitions.items())],
                      R2_reviews_preserved=sum(r['R2']['outcome']=='REVIEW_UNCERTAIN' and r['R2']==r['R3'] for r in records),
                      new_definitive_rejections=sum(r['R2']['outcome']!='REJECT_INVALID' and r['R3']['outcome']=='REJECT_INVALID' for r in records),
                      review_reasons=dict(Counter(c for r in records if r['R3']['outcome']=='REVIEW_UNCERTAIN' for c in set(r['R3']['reason_codes']))),
                      localization=runtime.localization.summarize(records),localization_by_parent={p:runtime.localization.summarize([r for r in records if r['source_parent_id']==p]) for p in dataset.groups},
                      ambiguous_ACCEPT=sum(r['R3']['outcome']=='ACCEPT' and r['ambiguous_text_alignment'] for r in records),carpet_generation_ready=0,
                      frozen_modules=runtime.loaded_sources(),policy_invariants_pass=True)
        for r in records:runtime.policy.assert_invariants(r['R2']['outcome'],r['R3']['outcome'])
        if measured['new_definitive_rejections']!=0:raise AssertionError('New text-only rejection')
        write_json(out/'metrics.json',measured)
        runtime.integration.write_csv(out/'outcomes.csv',[dict(image_id=r['image_id'],parent=r['source_parent_id'],validity=r['validity'],category=r['category'],variant=r['variant'],
                       R2=r['R2']['outcome'],R3=r['R3']['outcome'],review_reasons=r['R3']['reason_codes'],ambiguous_text_alignment=r['ambiguous_text_alignment'],carpet_generation_ready=False) for r in records])
        (out/'report.md').write_text('# Frozen R3 '+args.split+' results\n\nREVIEW is not definitive REJECT. R3 ACCEPT is not carpet-generation authorization. No tuning or R3.1 is used.\n\n~~~json\n'+json.dumps(summary,indent=2)+'\n~~~\n\nSee metrics.json for parent/category/variant results, transitions, reasons and localization coverage. Broad row highlights do not imply precise glyph localization.\n',encoding='utf-8')
        verify()
        names=('run_manifest.json','metrics.json','per_image.jsonl','outcomes.csv','report.md')
        write_json(out/'completion.json',dict(complete=True,kind='evaluation',split=args.split,policy_id='R3_HITL_V3',records=600,parents=150,release_sha256=release_sha,outputs={n:sha(out/n) for n in names}))
        write_json(state/'evaluation_completed.json',dict(split=args.split,run_directory=str(out),release_sha256=release_sha,completion_sha256=sha(out/'completion.json')))
        print(out/'report.md')


def compare(args):
    verify();valid=check_completion(args.validation_run,'evaluation');test=check_completion(args.protected_run,'evaluation')
    if valid['split']!='validation_calibration' or test['split']!='protected_test' or any(d['records']!=600 or d['parents']!=150 for d in (valid,test)):raise ValueError('Comparison requires the matching complete 600-image/150-parent validation and protected-test results')
    m=load(RELEASE/'release_manifest.json');development=load(RELEASE/'provenance/approved_development/metrics.json')['candidates']['R3_HITL_V3']['summary']
    validation=load(Path(args.validation_run)/'metrics.json')['summary'];protected=load(Path(args.protected_run)/'metrics.json')['summary']
    out=output_path(args.output_dir);out.mkdir(parents=True,exist_ok=False)
    result=dict(policy_id='R3_HITL_V3',release_sha256=sha(RELEASE/'release_manifest.json'),development=development,validation=validation,protected_test=protected,
                validation_completion_sha256=sha(Path(args.validation_run)/'completion.json'),protected_completion_sha256=sha(Path(args.protected_run)/'completion.json'),
                limitations='Synthetic fixed-grid Talim; three mutation categories; REVIEW is not definitive REJECT; OCR can normalize wrong-tier glyphs; no expert validation or physical carpet alignment established')
    write_json(out/'comparison.json',result)
    lines=['# Final Stage 2F comparison','','Selected policy: R3_HITL_V3. R3.1 experiments are preserved and excluded. REVIEW is not definitive rejection.','','| Split | Valid A/R/J | Invalid A/R/J | Invalid ACCEPT rate | Valid REVIEW rate | False definitive REJECT rate |','|---|---|---|---|---|---|']
    for name,s in [('Development',development),('Validation',validation),('Protected test',protected)]:
        counts=s['three_way_confusion'];fmt=lambda v:'/'.join(str(counts[v][o]) for o in ('ACCEPT','REVIEW_UNCERTAIN','REJECT_INVALID'))
        lines.append('| '+name+' | '+fmt('VALID')+' | '+fmt('INVALID')+' | '+str(s['invalid_accepted_rate'])+' | '+str(s['valid_review_rate'])+' | '+str(s['false_definitive_rejection_rate'])+' |')
    lines+=['','A/R/J means ACCEPT / REVIEW_UNCERTAIN / REJECT_INVALID. Report declines honestly; no tuning after held-out exposure.','',result['limitations'],'','Detailed localization/category/parent evidence remains in the sealed per-split metrics. Carpet generation requires separate alignment resolution.']
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write_json(out/'completion.json',dict(complete=True,kind='comparison',release_sha256=result['release_sha256'],outputs={n:sha(out/n) for n in ('comparison.json','report.md')}));print(out/'report.md')


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    for name in ('preflight','infer','evaluate'):
        s=sub.add_parser(name);s.add_argument('--split',choices=SPLITS,required=True)
        if name!='preflight':
            s.add_argument('--authorization',required=True);s.add_argument('--validation-run',type=Path);s.add_argument('--ack-validation-reviewed',action='store_true');s.add_argument('--output-dir',type=Path,required=True)
        if name=='infer':s.add_argument('--resume',action='store_true')
        if name=='evaluate':s.add_argument('--raw-dir',type=Path,required=True)
    s=sub.add_parser('compare');s.add_argument('--validation-run',type=Path,required=True);s.add_argument('--protected-run',type=Path,required=True);s.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();globals()[a.command](a)


if __name__=='__main__':main()
