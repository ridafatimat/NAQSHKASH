"""Offline spatial feasibility. No model imports, inference, decisions or fitting."""
import argparse,json,sys,math
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
WORKSPACE=Path(r'C:\NAQSHKASH\stage2\stage2f_rejection_evaluation_v2')
HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(WORKSPACE/'src'))
from integration import load_json,digest,write_json,rules,content_hash
from dataset_loader import DevelopmentDataset
from evaluate_r2 import verify_r2
from r3_fidelity_diagnostic import normalized
from measurements import alignment_targets
sys.path.insert(0,str(WORKSPACE/'r3_alignment_experiment/src'))
from alignment import associate,project


def latest(root,required):
    found=[]
    for p in Path(root).iterdir():
        if p.is_dir() and (p/'completion.json').is_file():
            done=load_json(p/'completion.json')
            if done.get('complete') and all((p/n).is_file() for n in required):found.append(p)
    if not found:raise ValueError('No completed run at '+str(root))
    return max(found,key=lambda p:((p/'completion.json').stat().st_mtime_ns,p.name))


def inputs():
    verify_r2();dataset=DevelopmentDataset(verify_files=False)
    stage=latest(WORKSPACE/'r3_text_alignment_experiment/runs',('metrics.json','report.md','per_image.jsonl'))
    done=load_json(stage/'completion.json')
    if done['metrics_sha256']!=digest(stage/'metrics.json') or done['evidence_sha256']!=digest(stage/'per_image.jsonl'):raise ValueError('Stage 1 changed')
    metrics=load_json(stage/'metrics.json');rows=[json.loads(x) for x in (stage/'per_image.jsonl').read_text().splitlines()]
    if len(rows)!=2800 or {r['image_id'] for r in rows}!=set(dataset.by_id):raise ValueError('Not the complete development set')
    provenance=metrics['provenance'];r2path=Path(provenance['r2_dir'])/'results.json'
    if digest(r2path)!=provenance['r2_results_sha256']:raise ValueError('R2 source changed')
    r2={r['image_id']:r for r in load_json(r2path)['rows']}
    for r in rows:
        m=dataset.by_id[r['image_id']]
        if r['parent']!=m['source_parent_id'] or r['validity']!=m['validity_status'] or r['category']!=m['invalidity_category']:raise ValueError('Stage 1 labels/identity drift')
    ctc=latest(WORKSPACE/'r3_alignment_experiment/runs',('metrics.json','report.md','per_image.json'))
    cd=load_json(ctc/'completion.json')
    if cd['metrics_sha256']!=digest(ctc/'metrics.json'):raise ValueError('CTC evaluation changed')
    extractions=[p for p in (WORKSPACE/'r3_alignment_experiment/runs').iterdir() if p.is_dir() and (p/'observations.jsonl').is_file() and digest(p/'observations.jsonl')==cd['source_run_sha256']]
    if len(extractions)!=1:raise ValueError('Cannot identify CTC extraction')
    extraction=extractions[0];ed=load_json(extraction/'completion.json')
    if not ed['complete'] or ed['raw_sha256']!=cd['source_run_sha256'] or ed['plan_sha256']!=digest(extraction/'plan.json'):raise ValueError('CTC extraction incomplete')
    cached=load_json(ctc/'per_image.json');sealed={x['identity']['image_id']:x for x in map(json.loads,(extraction/'observations.jsonl').read_text().splitlines())}
    if len(cached)!=48 or len(sealed)!=48 or {x['image_id'] for x in cached}!=set(sealed):raise ValueError('CTC pilot coverage differs')
    for c in cached:
        if c['image_id'] not in dataset.by_id or c['capture']!=sealed[c['image_id']]['observation']:raise ValueError('CTC cache identity/capture mismatch')
        item=sealed[c['image_id']]
        if item['payload_sha256']!=content_hash({k:v for k,v in item.items() if k!='payload_sha256'}) or item['identity']!=dataset.identity(dataset.by_id[c['image_id']]):raise ValueError('CTC record proof/identity drift')
    paths=[stage/n for n in ('completion.json','metrics.json','report.md','per_image.jsonl')]+[r2path]+[ctc/n for n in ('completion.json','metrics.json','report.md','per_image.json')]+[extraction/n for n in ('completion.json','plan.json','observations.jsonl')]
    return dataset,rows,r2,cached,extraction,dict(stage1=str(stage),ctc_evaluation=str(ctc),ctc_extraction=str(extraction),sha256={str(p):digest(p) for p in paths})


def n(group):return dict(images=len(group),parents=len({g.get('parent',g.get('parent_id')) for g in group}))


def audit():
    from talim_validator import validate_aligned_row
    ds,stage,r2,ctc,extraction,provenance=inputs();cached_ids={x['image_id'] for x in ctc};valid=[];remaining=[]
    for a in stage:
        m=ds.by_id[a['image_id']];pred=r2[a['image_id']]['interpreted_transcription'];ref=m['raw_display']
        if a['validity']=='VALID' and a['analysis']['conditional_text_contradiction']:
            pr=normalized(pred,ink=True,aliases=True);rr=normalized(ref,ink=True,aliases=True)
            role_bad={role for i in range(max(len(pr),len(rr))) for role in ('upper','count','lower') if (pr[i].get(role,'') if i<len(pr) else '')!=(rr[i].get(role,'') if i<len(rr) else '')}
            only_space=pr==rr;pattern=normalized(pred,aliases=True)==normalized(ref,aliases=True)
            total_only=all(z['passing_count'] for z in a['analysis']['rows'])
            whitespace_differences=[]
            for i,row in enumerate(ref):
                expected={('count' if k=='count_line' else k):v.strip(' ') for k,v in row.items() if v.strip(' ')}
                actual=pred[i] if i<len(pred) else {}
                for role in set(expected)|set(actual):
                    x=expected.get(role,'');y=actual.get(role,'')
                    if x!=y and x.replace(' ','')==y.replace(' ',''):whitespace_differences.append(dict(row=i,role=role,reference=x,prediction=y))
            refs_ok=[validate_aligned_row(row['upper'],row['count_line'],row['lower'])['outcome']=='ACCEPT' for row in ref]
            cause='space_pattern_error_only' if only_space else 'count_glyph_error' if 'count' in role_bad else 'symbol_glyph_error'
            if pattern:cause='unresolved_reconstruction_limitation'
            valid.append(dict(image_id=a['image_id'],parent=a['parent'],new_review=a['original_R2_outcome']=='ACCEPT',cause=cause,
                role_glyph_errors=sorted(role_bad),space_errors=whitespace_differences,row_total_conflict_only=total_only,
                canonical_source_rows_pass=all(refs_ok),cached=a['image_id'] in cached_ids,
                row_failures=[z['failure_reasons'] for z in a['analysis']['rows'] if not z['passing_count']],
                row_totals=[z['row_totals'] for z in a['analysis']['rows']],
                source_row_totals=[sum(run['count'] for run in row['runs']) for row in m['original_structured_notation']['logical_rows']]))
        if a['validity']=='INVALID' and a['review_only_counterfactual']=='ACCEPT':
            p=m['mutation_parameters'];ri=p['row'];role='count' if p['target_tier']=='count_line' else p['target_tier']
            before=ref[ri][p['target_tier']];after=pred[ri].get(role,'')
            target_options=[]
            if m['invalidity_category']=='wrong_tier':
                t=before.replace(' ','');v=after.replace(' ','');idx=len(before[:p['column']].replace(' ',''))
                target_options=sorted(alignment_targets(t,v,idx),key=str)
            control=next(z for z in ds.groups[a['parent']] if z['validity_status']=='VALID' and z['augmentation_recipe']==m['augmentation_recipe'])
            ctext=ref[ri]['count_line'];where=('leading' if not ctext[:p['column']].strip() else 'trailing' if not ctext[p['column']+p['width']:].strip() else 'internal')
            remaining.append(dict(image_id=a['image_id'],parent=a['parent'],category=a['category'],variant=a['variant'],
                cached=a['image_id'] in cached_ids,source_glyph=p['original_symbol'],target_tier=p['target_tier'],predicted_alternatives=target_options,
                removed_token=p['original_token'],reversed_or_missing_location=where,
                literal_mutated_tier=before.strip(' '),predicted_tier=after,
                matched_control=control['image_id'],control_text_identical=pred==r2[control['image_id']]['interpreted_transcription'],
                pattern_faithful=a['offline_scoring']['pattern_faithful'],glyph_faithful=a['offline_scoring']['glyph_faithful'],
                page_candidates=a['analysis']['page_candidate_count_in_search'],common_totals=a['analysis']['common_row_totals'],
                example_witness=[z['successful_candidates'][0] for z in a['analysis']['rows']]))
    # Read cached positional results, not a new coordinate correction or rescore.
    errors=defaultdict(list);shape_bad=[];matrix_types=Counter();oracle_correct=oracle_denom=0;oracle_risk=set();first_error=defaultdict(list)
    first_emissions=Counter();oracle_errors=Counter()
    for c in ctc:
        cap=c['capture'];M=np.asarray(cap['source_to_processed_matrix']);matrix_types['identity' if np.array_equal(M,np.eye(3)) else 'nonidentity']+=1
        refs=c['score']['reference']
        for tier in cap['tiers']:
            x0,y0,x1,y1=tier['box'];wr,hr=tier['resized_dimensions']
            if wr!=round((x1-x0)*(64/float(y1-y0))) or hr!=64 or tier['frames']!=wr//4:shape_bad.append(c['image_id'])
            visible=[s for s in tier['characters'] if s['character']!=' ']
            if visible:
                first_emissions['tiers']+=1;first_emissions['first_visible_span_starts_frame_zero']+=visible[0]['start_frame']==0
        for role,gs in c['score']['predictions'].items():
            for j,g in enumerate(gs):
                idx=g['reference_index']
                if idx is not None and g.get('mapping'):
                    err=g['mapping']['source_center'][0]-refs[role][idx]['x'];errors[role].append(err)
                    first_error['first' if idx==0 else 'later'].append(err)
                    if idx==0:first_emissions['first_mapped_point_before_reference_ink_left']+=g['mapping']['source_center'][0]<refs[role][idx]['box'][0]-.5
        # OFFLINE oracle control: same association algorithm at true ink centers.
        symbols=sorted([dict(x=g['x'],role=role,index=i,column=g['column']) for role in ('upper','lower') for i,g in enumerate(refs[role])],key=lambda g:g['x'])
        counts=[dict(x=g['x'],column=g['column']) for g in refs['count']];groups=associate(symbols,counts)
        if groups['status']=='unique':
            starts=sorted([(s['column'],i) for i,s in enumerate(symbols)])
            for si,group in enumerate(groups['groups']):
                start=symbols[si]['column'];end=min([col for col,j in starts if col>start],default=10**9)
                for ci in group:
                    ok=start<=counts[ci]['column']<end;oracle_correct+=ok;oracle_denom+=1
                    if not ok:oracle_errors[c['category']]+=1
                    if not ok and c['validity']=='VALID':oracle_risk.add(c['image_id'])
    def distribution(xs):return dict(count=len(xs),mean=float(np.mean(xs)),median=float(np.median(xs)),min=float(min(xs)),max=float(max(xs))) if xs else None
    output=dict(provenance=provenance,valid=valid,remaining=remaining,valid_counts=n(valid),remaining_counts=n(remaining),
        valid_new_counts=n([v for v in valid if v['new_review']]),valid_causes={cause:n([v for v in valid if v['cause']==cause]) for cause in sorted({v['cause'] for v in valid})},
        new_valid_causes={cause:n([v for v in valid if v['cause']==cause and v['new_review']]) for cause in sorted({v['cause'] for v in valid})},
        valid_total_conflicts=n([v for v in valid if v['row_total_conflict_only']]),valid_with_space_flags=n([v for v in valid if v['space_errors']]),
        remaining_categories={k:n([v for v in remaining if v['category']==k]) for k in sorted({v['category'] for v in remaining})},
        cached_valid_contradictions=n([v for v in valid if v['cached']]),cached_remaining=n([v for v in remaining if v['cached']]),
        mapping=dict(matrix_types=dict(matrix_types),dimension_or_frame_mismatch=shape_bad,bias_by_role={k:distribution(v) for k,v in errors.items()},
            first_emissions=dict(first_emissions),oracle_errors_by_category=dict(oracle_errors),
            first_vs_later={k:distribution(v) for k,v in first_error.items()},
            oracle_true_ink_center_association=dict(correct=oracle_correct,denominator=oracle_denom,valid_controls_with_errors=len(oracle_risk),
                scope='offline association-algorithm control only; not a coordinate correction or deployable result')))
    return output


def ink_groups(pixels):
    """Runtime-style pixel diagnostic: no text, font, annotation or CTC input.

    Merge multi-component ink only where x projections touch/overlap. Separated
    dot/bar fragments remain unresolved, not asserted to be separate glyphs.
    No size cutoff, learned cell pitch or gap threshold is used.
    """
    import cv2
    a=np.asarray(pixels)
    if a.ndim!=2 or a.dtype!=np.uint8 or not a.size:raise ValueError('Expected uint8 grayscale crop')
    if a.min()==a.max():return dict(status='constant_unreadable_or_blank',groups=[],components=[],glyph_identities='NOT_INFERRED')
    threshold,mask=cv2.threshold(a,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    count,labels,stats,centers=cv2.connectedComponentsWithStats(mask,connectivity=8)
    comps=[dict(component=i,box=[int(x),int(y),int(x+w),int(y+h)],area=int(area)) for i,(x,y,w,h,area) in enumerate(stats[1:],1)]
    active=np.any(mask>0,axis=0);edges=np.diff(np.pad(active.astype(np.int8),(1,1)))
    groups=[]
    for left,right in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
        ys,xs=np.where(mask[:,left:right]>0);members=[c for c in comps if c['box'][0]<right and c['box'][2]>left]
        groups.append(dict(box=[int(left),int(ys.min()),int(right),int(ys.max()+1)],components=[c['component'] for c in members],
            component_count=len(members),foreground_pixels=int(len(xs))))
    return dict(status='observed_ink',threshold_method='per-crop Otsu; no sweep or label calibration',threshold=float(threshold),
        groups=groups,components=comps,glyph_identities='NOT_INFERRED')


def geometry_associations(tiers):
    """Candidate corridors from IMAGE x-groups only. Not a Talim validator."""
    symbols=sorted([dict(g,role=role,index=i) for role in ('upper','lower') for i,g in enumerate(tiers.get(role,{}).get('groups',[]))],key=lambda g:g['global_box'][0])
    counts=tiers.get('count',{}).get('groups',[])
    if any(symbols[i]['global_box'][0]==symbols[i-1]['global_box'][0] for i in range(1,len(symbols))):
        return dict(status='ambiguous_equal_symbol_anchors',symbols=symbols,assignments=[],empty_corridors=[])
    assignments=[]
    for ci,g in enumerate(counts):
        center=(g['global_box'][0]+g['global_box'][2]-1)/2
        eligible=[i for i,s in enumerate(symbols) if s['global_box'][0]<=center]
        assignments.append(dict(count_group=ci,symbol_group=eligible[-1] if eligible else None))
    occupied={a['symbol_group'] for a in assignments if a['symbol_group'] is not None}
    return dict(status='candidate_corridors' if symbols else 'no_symbol_anchors',symbols=symbols,assignments=assignments,
        empty_corridors=[i for i in range(len(symbols)) if i not in occupied],
        caveat='An ink group can be a fragment/merge. Empty corridor is not confirmed missing count.')


def component_observation(capture,read_crop):
    """Only cached crop pixels + D1 geometry; no renderer or truth in this API."""
    result={}
    for t in capture['tiers']:
        measured=ink_groups(read_crop(t));x0,y0,_,_=t['box']
        for g in measured['groups']:
            a,b,c,d=g['box'];g['global_box']=[a+x0,b+y0,c+x0,d+y0]
        result[t['role']]=measured
    return dict(tiers=result,corridors=geometry_associations(result))


def score_components(observed,case):
    """OFFLINE references enter after pixel-only observation/corridor decisions."""
    refs=case['score']['reference'];M=np.asarray(case['capture']['source_to_processed_matrix']);inverse=np.linalg.inv(M)
    matches={};metrics=dict(reference_glyphs=sum(map(len,refs.values())),matched_glyphs=0,fragmented_reference_glyphs=0,
        merged_or_unassigned_groups=0,position_errors=[],normalized_position_errors=[],association_correct=0,association_denominator=0,
        visible_count_denominator=len(refs['count']),expected_empty_tokens=0,detected_empty_tokens=0,
        false_empty_tokens=0,empty_candidates_unscorable=0,wrong_tier_identification='NOT_ESTABLISHED: positioning is not character recognition')
    for role,reference in refs.items():
        boxes=[]
        for g in reference:
            a,b,c,d=g['box'];p=project([(a-.5,b-.5),(c-.5,b-.5),(c-.5,d-.5),(a-.5,d-.5)],M)
            boxes.append([min(v[0] for v in p),max(v[0] for v in p)])
        groups=observed['tiers'].get(role,{}).get('groups',[]);options=[];owners=Counter()
        for g in groups:
            left,top,right,bottom=g['global_box'];ids=[i for i,(a,b) in enumerate(boxes) if max(left-.5,a)<min(right-.5,b)]
            options.append(ids)
            if len(ids)==1:owners[ids[0]]+=1
        metrics['fragmented_reference_glyphs']+=sum(count>1 for count in owners.values())
        match=[]
        for g,ids in zip(groups,options):
            index=ids[0] if len(ids)==1 and owners[ids[0]]==1 else None;match.append(index)
            if index is None:metrics['merged_or_unassigned_groups']+=1
            else:
                left,top,right,bottom=g['global_box'];point=project([((left+right-1)/2,(top+bottom-1)/2)],inverse)[0]
                error=point[0]-reference[index]['x'];metrics['matched_glyphs']+=1;metrics['position_errors'].append(error)
                metrics['normalized_position_errors'].append(abs(error)/reference[index]['width'])
        matches[role]=match
    symbols=observed['corridors']['symbols'];truth_symbols=sorted([(g['column'],role,i) for role in ('upper','lower') for i,g in enumerate(refs[role])]);true_counts={};count_owner={}
    for j,(start,role,idx) in enumerate(truth_symbols):
        end=truth_symbols[j+1][0] if j+1<len(truth_symbols) else 10**9
        inds=[i for i,g in enumerate(refs['count']) if start<=g['column']<end];true_counts[role,idx]=inds
        if not inds:metrics['expected_empty_tokens']+=1
        for i in inds:count_owner[i]=(role,idx)
    for assignment in observed['corridors']['assignments']:
        ci=assignment['count_group'];si=assignment['symbol_group'];ri=matches['count'][ci]
        if si is None:continue
        s=symbols[si];sr=matches[s['role']][s['index']]
        if ri is not None and sr is not None:
            metrics['association_denominator']+=1;metrics['association_correct']+=count_owner[ri]==(s['role'],sr)
    for si in observed['corridors']['empty_corridors']:
        s=symbols[si];sr=matches[s['role']][s['index']]
        if sr is None:metrics['empty_candidates_unscorable']+=1
        elif not true_counts[s['role'],sr]:metrics['detected_empty_tokens']+=1
        else:metrics['false_empty_tokens']+=1
    return dict(metrics=metrics,offline_group_reference_matches=matches)


def components(output):
    from PIL import Image
    ds,stage,r2,cases,run,provenance=inputs();out=Path(output).resolve()
    if not out.is_relative_to((HERE/'runs').resolve()) or out.exists():raise ValueError('Use a NEW directory under this diagnostic/runs')
    scored=[]
    remaining_ids={r['image_id'] for r in stage if r['validity']=='INVALID' and r['review_only_counterfactual']=='ACCEPT'}
    contradiction_ids={r['image_id'] for r in stage if r['validity']=='VALID' and r['analysis']['conditional_text_contradiction']}
    for case in cases:
        cap=case['capture'];base=(run/cap['asset_directory']).resolve()
        if not base.is_relative_to(run.resolve()):raise ValueError('Unsafe asset directory')
        def read(t):
            p=(base/t['crop_file']).resolve()
            if not p.is_relative_to(base) or digest(p)!=t['crop_sha256']:raise ValueError('Cached crop drift')
            with Image.open(p) as im:return np.asarray(im.convert('L'))
        observed=component_observation(cap,read);score=score_components(observed,case)
        scored.append(dict(image_id=case['image_id'],parent=case['parent_id'],validity=case['validity'],category=case['category'],variant=case['variant'],observed=observed,score=score,
            stage1_remaining_accept=case['image_id'] in remaining_ids,stage1_valid_contradiction=case['image_id'] in contradiction_ids))
    def aggregate(group):
        keys=('reference_glyphs','matched_glyphs','fragmented_reference_glyphs','merged_or_unassigned_groups','association_correct','association_denominator','visible_count_denominator','expected_empty_tokens','detected_empty_tokens','false_empty_tokens','empty_candidates_unscorable')
        m={k:sum(c['score']['metrics'][k] for c in group) for k in keys};errors=[e for c in group for e in c['score']['metrics']['position_errors']]
        norm=[e for c in group for e in c['score']['metrics']['normalized_position_errors']]
        m.update(images=len(group),parents=len({c['parent'] for c in group}),position_mean=float(np.mean(errors)) if errors else None,
            position_median_absolute=float(np.median(np.abs(errors))) if errors else None,
            position_p95_absolute=float(np.percentile(np.abs(errors),95)) if errors else None,
            normalized_position_median=float(np.median(norm)) if norm else None,normalized_position_p95=float(np.percentile(norm,95)) if norm else None,
            valid_controls_with_false_empty=sum(c['validity']=='VALID' and c['score']['metrics']['false_empty_tokens']>0 for c in group),
            valid_controls_with_wrong_association=sum(c['validity']=='VALID' and c['score']['metrics']['association_correct']<c['score']['metrics']['association_denominator'] for c in group),
            controls_with_incomplete_glyph_grouping=sum(c['validity']=='VALID' and c['score']['metrics']['matched_glyphs']<c['score']['metrics']['reference_glyphs'] for c in group),
            wrong_tier_identification_accuracy=None)
        def ratio(a,b):return a/b if b else None
        m.update(alignment_coverage=ratio(m['matched_glyphs'],m['reference_glyphs']),
            association_accuracy_conditional=ratio(m['association_correct'],m['association_denominator']),
            association_scoring_coverage=ratio(m['association_denominator'],m['visible_count_denominator']),
            empty_token_detection_recall=ratio(m['detected_empty_tokens'],m['expected_empty_tokens']))
        return m
    result=dict(provenance=provenance,overall=aggregate(scored),by_validity={v:aggregate([c for c in scored if c['validity']==v]) for v in ('VALID','INVALID')},
        by_category={v:aggregate([c for c in scored if c['category']==v]) for v in sorted({c['category'] for c in scored})},
        by_variant={v:aggregate([c for c in scored if c['variant']==v]) for v in ('clean','illuminated')},
        by_parent={v:aggregate([c for c in scored if c['parent']==v]) for v in sorted({c['parent'] for c in scored})},
        targeted_cohorts={k:aggregate([c for c in scored if c[k]]) for k in ('stage1_remaining_accept','stage1_valid_contradiction')},
        limitations='No semantic glyph identification or rejection policy. Zero overlap with the 70 valid Stage 1 contradictions. Conditional grouping matches must be read with full coverage.',
        historical_CTC_metrics=load_json(Path(provenance['ctc_evaluation'])/'metrics.json')['overall'])
    out.mkdir(parents=True,exist_ok=False);write_json(out/'metrics.json',result);write_json(out/'per_image.json',scored)
    (out/'report.md').write_text('# Cached-crop component feasibility\n\nNo OCR, coordinate correction or rejection policy. Runtime-style groups/corridors use only crop pixels and D1 boxes. References score afterward. Group fragments/merges are unscorable, not silently fused using annotations.\n\n~~~json\n'+json.dumps(result['overall'],indent=2)+'\n~~~\n\nCompare association accuracy and full-denominator coverage with historical_CTC_metrics. Inspect valid-control false-empty and incomplete-grouping counts before any safety claim. Glyph identity is not inferred, so wrong-tier identification accuracy is null. This 12-parent cache has no crops for the 70 valid contradictions; it cannot establish a reduction in their REVIEW burden.\n',encoding='utf-8')
    write_json(out/'completion.json',dict(complete=True,images=len(scored),metrics_sha256=digest(out/'metrics.json')));verify_r2();print(out/'report.md')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('audit','components'));parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args()
    if args.command=='audit':print(json.dumps(audit(),indent=2,ensure_ascii=False))
    elif args.output_dir is None:parser.error('components requires --output-dir')
    else:components(args.output_dir)
