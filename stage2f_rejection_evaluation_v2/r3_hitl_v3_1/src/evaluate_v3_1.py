"""Manual-only development evaluation. D1 pixels only; no E1 initialization."""
import argparse,copy,json,platform,sys,uuid
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
from importlib.metadata import version
from paths_v3_1 import HERE,R3,WORKSPACE,ID,config,verify_locks,own_hashes,load_json,digest,content_hash,write_json,write_csv
from evaluate_v3 import verify_completed as verify_r3_run,load_sources
from policy_v3 import load_policy,runtime_inputs
from evaluate_r2 import verify_r2
from dataset_loader import image_bytes
from evaluation_metrics import grouped
from policy_v3_1 import decide,assert_invariants
from ink_measurement import D1Only
from offline_scoring import score,summary as scoring_summary


def output_path(path):
    p=Path(path).resolve()
    if not p.is_relative_to((HERE/'runs').resolve()):raise ValueError('Outputs must stay in r3_hitl_v3_1/runs')
    return p


def load_inputs(c):
    verify_locks(c);verify_r2()
    run,manifest,done=verify_r3_run(c['r3_run'])
    if done['images']!=2800 or done['parents']!=700 or manifest['split']!='development_design':raise ValueError('Not exact authorized development run')
    for p,h in manifest['source_code_and_protocol_sha256'].items():
        if digest(p)!=h:raise ValueError('Current R3 differs from measured R3 source: '+p)
    r3config=load_policy()
    if r3config!=manifest['config']:raise ValueError('R3 config changed after its completed run')
    dataset,bases,raw,stage1,lineage=load_sources(r3config,manifest['source']['r2_dir'],manifest['source']['stage1_dir'])
    if dataset.manifest_sha256!=manifest['source']['dataset_manifest_sha256'] or lineage!=manifest['source']:raise ValueError('R3/R2/dataset lineage mismatch')
    records=[json.loads(line) for line in (run/'per_image.jsonl').read_text(encoding='utf-8').splitlines()]
    if len(records)!=2800 or len({r['image_id'] for r in records})!=2800 or {r['image_id'] for r in records}!=set(dataset.by_id):raise ValueError('R3 development coverage differs')
    byid={r['image_id']:r for r in records}
    for b in bases:
        r=byid[b['image_id']];src=dataset.by_id[b['image_id']];ev=r['alignment_evidence']
        if (r['split']!='development_design' or r['policy_id']!='R3_HITL_V3' or r['image_sha256']!=src['image_sha256'] or r['source_parent_id']!=src['source_parent_id'] or
            r['interpreted_transcription']!=b['interpreted_transcription'] or r['R2']!=b['decision'] or ev['analysis']!=stage1[b['image_id']]['analysis'] or
            ev['input_sha256']!=content_hash(runtime_inputs(b['interpreted_transcription'],raw[b['image_id']],r3config)) or
            ev['engine_sha256']!=digest(WORKSPACE/'r3_text_alignment_experiment/src/text_alignment.py') or ev['max_offset']!=r3config['max_offset'] or ev['max_candidates']!=r3config['max_candidates']):
            raise ValueError('Saved R3 evidence/text/identity binding failed: '+b['image_id'])
        from policy_v3 import assert_invariants as r3_invariant
        r3_invariant(r['R2']['outcome'],r['R3']['outcome'])
    return dataset,bases,raw,byid,dict(r3_run=str(run),r3_completion_sha256=digest(run/'completion.json'),R3_manifest_sha256=digest(run/'run_manifest.json'),**lineage)


def process_case(prior,raw,measurement,source):
    # Runtime whitelist ONLY. The policy cannot access source or prior scoring.
    result=decide(prior['R3'],prior['interpreted_transcription'],prior['alignment_evidence']['analysis'],measurement)
    assert_invariants(prior['R3']['outcome'],result)
    # Preserve old findings/overlays/localization verbatim alongside new evidence.
    r=dict(image_id=prior['image_id'],source_parent_id=prior['source_parent_id'],image_sha256=prior['image_sha256'],split='development_design',
           prior_R3=copy.deepcopy(prior['R3']),prior_R3_findings=copy.deepcopy(prior['findings']),prior_review_regions=copy.deepcopy(prior['review_regions']),
           prior_localization=copy.deepcopy(prior['localization']),R3_1=result,measurement=measurement)
    # OFFLINE ONLY: no feedback into the finalized decision above.
    r.update(validity=source['validity_status'],category=source['invalidity_category'],variant='illuminated' if source['augmentation_recipe'] else 'clean')
    r['offline_score']=score(result,measurement,source,prior['interpreted_transcription'])
    return r


def metrics(records,bases):
    byid={r['image_id']:r for r in records};versions={}
    for v in ('R3','R3_1'):
        rows=[]
        for b in bases:
            item=copy.deepcopy(b);r=byid[b['image_id']];d=r['prior_R3'] if v=='R3' else r['R3_1']['decision']
            item.update(actual_outcome=d['outcome'],triggered_reason_codes=d['reason_codes']);rows.append(item)
        total,parents,groups=grouped(rows)
        total['outcome_counts_and_rates']={validity:{o:dict(count=n,rate=n/sum(counts.values())) for o,n in counts.items()} for validity,counts in total['three_way_confusion'].items() if sum(counts.values())}
        versions[v]=dict(summary=total,parents=parents,groups=groups)
    changes=[dict(image_id=r['image_id'],parent=r['source_parent_id'],validity=r['validity'],category=r['category'],variant=r['variant'],
                  R3=r['prior_R3']['outcome'],R3_1=r['R3_1']['decision']['outcome'],reasons=[f['reason_code'] for f in r['R3_1']['findings']],
                  before_readings=r['R3_1']['distinct_readings_before'],after_readings=r['R3_1']['distinct_readings_after'],measurement_reason=(r['measurement'] or {}).get('reason'))
             for r in records if r['prior_R3']['outcome']!=r['R3_1']['decision']['outcome']]
    accepted=[r for r in records if r['prior_R3']['outcome']=='ACCEPT'];measured=[r for r in records if (r['measurement'] or {}).get('status')!='not_attempted']
    positions=[r for r in measured if r['measurement'] and r['measurement'].get('status')=='complete']
    transitions=Counter((r['prior_R3']['outcome'],r['R3_1']['decision']['outcome']) for r in records)
    from offline_scoring import ratio
    scoring=scoring_summary(records);saved_tiers=sum(r.get('saved_D1_tiers',0) for r in records)
    attempted_tiers=sum(r.get('saved_D1_tiers',0) for r in measured)
    scoring.update(saved_D1_tier_denominator=saved_tiers,attempted_saved_D1_tiers=attempted_tiers,
                   first_ink_coverage_all_saved_tiers=ratio(scoring['first_ink_measured'],saved_tiers),
                   first_ink_coverage_attempted_saved_tiers=ratio(scoring['first_ink_measured'],attempted_tiers),
                   offset_scoring_coverage_all_saved_tiers=ratio(scoring['offset_eligible'],saved_tiers),
                   exact_offset_full_saved_tier_rate=ratio(scoring['exact_offsets'],saved_tiers))
    return dict(candidates=versions,R3_to_R3_1_transitions=[dict(before=a,after=b,images=n) for (a,b),n in sorted(transitions.items())],R3_differences=changes,
                policy_invariants_pass=True,new_definitive_rejections=sum(r['prior_R3']['outcome']!='REJECT_INVALID' and r['R3_1']['decision']['outcome']=='REJECT_INVALID' for r in records),
                R3_reviews_preserved=sum(r['prior_R3']['outcome']=='REVIEW_UNCERTAIN' and r['R3_1']['decision']==r['prior_R3'] for r in records),
                measurement_attempted=len(measured),measurement_complete=len(positions),D1_box_equality=dict(Counter(str((r['measurement'] or {}).get('box_equality')) for r in measured)),
                measured_tiers=sum(len(row['tiers']) for r in measured for row in (r['measurement'] or {}).get('rows',[])),
                saved_D1_tier_denominator=sum(r.get('saved_D1_tiers',0) for r in records),
                distinct_reading_counts_before=dict(Counter(str(r['R3_1']['distinct_readings_before']) for r in accepted)),
                distinct_reading_counts_after=dict(Counter(str(r['R3_1']['distinct_readings_after']) for r in accepted)),
                ambiguous_cases_resolved=sum(r['R3_1']['ambiguity_resolved'] for r in records),
                ambiguous_valid_cases_resolved=sum(r['validity']=='VALID' and r['R3_1']['ambiguity_resolved'] for r in records),
                ambiguous_remaining=sum((r['R3_1']['distinct_readings_after'] or 0)>1 or
                                        (not r['R3_1']['image_position_verified'] and (r['R3_1']['distinct_readings_before'] or 0)>1) for r in accepted),
                positioning_unavailable_on_prior_ACCEPT=sum(not r['R3_1']['image_position_verified'] for r in accepted),
                carpet_generation_ready=sum(r['R3_1']['carpet_generation_ready'] for r in records),
                carpet_generation_ready_rate=sum(r['R3_1']['carpet_generation_ready'] for r in records)/len(records),
                carpet_generation_ready_among_ACCEPT=ratio(sum(r['R3_1']['carpet_generation_ready'] for r in records),sum(r['R3_1']['decision']['outcome']=='ACCEPT' for r in records)),
                selected_reading_status=dict(Counter(r['R3_1']['selected_reading_status'] for r in records)),
                unavailable_reasons=dict(Counter((r['measurement'] or {}).get('reason','UNSPECIFIED') for r in measured if (r['measurement'] or {}).get('status')!='complete')),
                reason_counts=dict(Counter(f['reason_code'] for r in records for f in r['R3_1']['findings'])),
                scoring=scoring,scoring_by_parent={p:scoring_summary([r for r in records if r['source_parent_id']==p]) for p in sorted({r['source_parent_id'] for r in records})})


def evaluate(args):
    c=config();ds,bases,raw,prior,lineage=load_inputs(c)
    if args.preflight:
        print(json.dumps(dict(policy=ID,split='development_design',images=len(bases),parents=len(ds.groups),source=lineage,D1_only_supported=True,
                              pixel_reads=0,OCR_inference=False,full_evaluation_executed=False,measurement_scope=args.measure_scope),indent=2));return
    out=output_path(args.output_dir);out.mkdir(parents=True,exist_ok=False);initial=own_hashes()
    manifest=dict(policy_id=ID,split='development_design',run_identifier=out.name,created_UTC=datetime.now(timezone.utc).isoformat(),status='IN_PROGRESS',
                  completion_status_file='completion.json',source=lineage,images=len(bases),parents=len(ds.groups),configuration=c,own_source_sha256=initial,
                  measurement_scope=args.measure_scope,software=dict(python=platform.python_version(),numpy=version('numpy'),opencv=version('opencv-python'),Pillow=version('Pillow')),
                  restriction='Frozen D1 only. No recognizer construction or OCR inference; no held-out images.')
    write_json(out/'run_manifest.json',manifest)
    d1=D1Only();records=[]
    with (out/'per_image.jsonl').open('x',encoding='utf-8') as stream:
        for index,b in enumerate(bases,1):
            id=b['image_id'];previous=prior[id];observation=raw[id]
            measure=args.measure_scope=='all-development' or previous['R3']['outcome']=='ACCEPT'
            if measure:
                measurement=d1.measure(image_bytes(ds.directory,ds.by_id[id]),observation['segmentation_rows'],previous['interpreted_transcription'],c)
                if measurement.get('provenance',{}).get('input_image_sha256') not in (None,previous['image_sha256']):raise ValueError('Measured image identity differs')
            else:measurement=dict(status='not_attempted',reason='Existing R3 REVIEW/REJECT cannot change; measurement not needed for policy',rows=[],box_equality=None)
            r=process_case(previous,observation,measurement,ds.by_id[id]);r['saved_D1_tiers']=sum(len(row) for row in observation['segmentation_rows'])
            records.append(r);stream.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n');stream.flush()
            if index%100==0:print('Completed',index,'/',len(bases),flush=True)
    measured=metrics(records,bases);write_json(out/'metrics.json',measured)
    write_csv(out/'R3_differences.csv',measured['R3_differences'])
    write_csv(out/'outcomes.csv',[dict(image_id=r['image_id'],parent=r['source_parent_id'],validity=r['validity'],category=r['category'],variant=r['variant'],
                                    R3=r['prior_R3']['outcome'],R3_1=r['R3_1']['decision']['outcome'],selected_reading_status=r['R3_1']['selected_reading_status'],
                                    carpet_generation_ready=r['R3_1']['carpet_generation_ready'],measurement_status=r['measurement']['status']) for r in records])
    write_json(out/'measurement_diagnostics.json',dict(images=[dict(image_id=r['image_id'],status=r['measurement']['status'],reason=r['measurement'].get('reason'),
                      box_equality=r['measurement'].get('box_equality'),pitch=r['measurement'].get('pitch'),candidate_trace=r['R3_1']['candidate_trace'],
                      position_unavailable_reason=r['R3_1'].get('position_unavailable_reason'),offline_offset_scores=r['offline_score']['tiers']) for r in records],
                      authoritative_full_measurements='per_image.jsonl',scope=args.measure_scope))
    excerpt={k:v for k,v in measured.items() if k not in ('candidates','R3_differences','scoring_by_parent')}
    excerpt['outcomes']={v:data['summary'] for v,data in measured['candidates'].items()}
    (out/'report.md').write_text('# R3.1 development results\n\nREVIEW is not definitive rejection. ACCEPT alone does not authorize carpet generation: inspect carpet_generation_ready and the explicit selected reading.\n\n'
        'Reference claims (16,434/16,442 exact offsets, 138 resolved valid pages, approximately 32 invalid accepts) were not used to choose parameters. Scope, coverage, unavailable evidence and all changed cases are reported below and in the CSV/JSON artifacts. No held-out evaluation or freeze was performed.\n\n~~~json\n'+json.dumps(excerpt,indent=2)+'\n~~~\n',encoding='utf-8')
    verify_locks(c);verify_r2()
    if own_hashes()!=initial:raise ValueError('R3.1 source changed during run')
    _,_,_,_,final_lineage=load_inputs(c)
    if final_lineage!=lineage:raise ValueError('Input lineage changed during run')
    names=('run_manifest.json','metrics.json','per_image.jsonl','outcomes.csv','R3_differences.csv','measurement_diagnostics.json','report.md')
    write_json(out/'completion.json',dict(complete=True,status='COMPLETE',policy_id=ID,split='development_design',images=len(records),parents=len(ds.groups),
                                        outputs={n:digest(out/n) for n in names},input_R3_completion_sha256=lineage['r3_completion_sha256']))
    print(out/'report.md')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preflight',action='store_true');p.add_argument('--output-dir',type=Path)
    p.add_argument('--split',choices=('development_design',),default='development_design')
    p.add_argument('--measure-scope',choices=('accepted','all-development'),default='accepted')
    a=p.parse_args()
    if not a.preflight and a.output_dir is None:p.error('Use --output-dir for the manual run')
    evaluate(a)


if __name__=='__main__':main()
