"""Monotonic post-R3 disambiguation. No dataset, image IO or truth parameters."""
import copy,json,math
from collections import defaultdict
from paths_v3_1 import ID,content_hash,validate_aligned_row


def reading_key(runs):
    return json.dumps(runs,sort_keys=True,separators=(',',':'),allow_nan=False)


def catalogue(decoded_rows,analysis):
    if not decoded_rows or len(decoded_rows)!=len(analysis.get('rows',[])):
        raise ValueError('Missing complete row candidates')
    if not analysis.get('all_rows_exhaustive_in_literal_model') or analysis.get('any_validator_uncertainty'):
        raise ValueError('Incomplete or uncertain Stage 1 evidence')
    checked=[]
    for text,row in zip(decoded_rows,analysis['rows']):
        if not row.get('coverage',{}).get('conditional_text_model_exhaustive') or row.get('uncertain_candidates'):
            raise ValueError('Incomplete row evidence')
        candidates=[]
        for c in row.get('successful_candidates',[]):
            if not isinstance(c.get('runs'),list) or not c['runs']:raise ValueError('BLOCKER: saved candidate lacks actual ordered runs')
            offsets=c['offsets']
            if set(offsets)!={'upper','count','lower'} or offsets['count']!=0:raise ValueError('Unsupported candidate offsets')
            for role in ('upper','count','lower'):
                if text.get(role,''):
                    if type(offsets[role]) is not int or offsets[role]<0:raise ValueError('Invalid offset')
                elif offsets[role] is not None:raise ValueError('Offset on absent tier')
            aligned={role:(' '*offsets[role]+text[role] if text.get(role) else '') for role in ('upper','count','lower')}
            result=validate_aligned_row(aligned['upper'],aligned['count'],aligned['lower'])
            if result['outcome']!='ACCEPT' or result['evidence']['runs']!=c['runs'] or sum(r['count'] for r in c['runs'])!=c['row_total']:
                raise ValueError('Candidate fails frozen canonical verification or saved runs disagree')
            candidates.append(copy.deepcopy(c))
        checked.append(candidates)
    return checked


def reading_space(rows):
    """Factorized exact count, avoiding exponential candidate enumeration."""
    totals=[{c['row_total'] for c in row} for row in rows]
    common=set.intersection(*totals) if totals else set();count=0;by_total={}
    for total in sorted(common):
        groups=[];ways=1
        for row in rows:
            g={}
            for c in row:
                if c['row_total']!=total:continue
                key=reading_key(c['runs'])
                if key not in g:g[key]=dict(runs=c['runs'],supporting_offsets=[])
                if c['offsets'] not in g[key]['supporting_offsets']:g[key]['supporting_offsets'].append(c['offsets'])
            groups.append(list(g.values()));ways*=len(g)
        by_total[str(total)]=groups;count+=ways
    selected=None
    if count==1:
        total=next(k for k,g in by_total.items() if all(len(r)==1 for r in g))
        selected=dict(row_total=int(total),logical_rows=[dict(logical_row_index=i,**r[0]) for i,r in enumerate(by_total[total])])
    return dict(distinct_readings=count,common_row_totals=sorted(common),factorized_readings=by_total,selected=selected)


def measurement_available(measurement,decoded_rows):
    if not measurement or measurement.get('status')!='complete':return False
    if (measurement.get('geometry_verified') is not True or measurement.get('box_equality') is not True or
        measurement.get('coordinate_system')!='D1_processed_pixel_centers' or measurement.get('decoded_rows_sha256')!=content_hash(decoded_rows)):
        return False
    if len(measurement.get('rows',[]))!=len(decoded_rows):return False
    for i,(r,text) in enumerate(zip(measurement['rows'],decoded_rows)):
        if r.get('logical_row_index')!=i or set(r['tiers'])!={role for role,t in text.items() if t}:return False
        for t in r['tiers'].values():
            value=t.get('measured_offset');interval=t.get('offset_interval')
            if not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:return False
            if not isinstance(interval,list) or len(interval)!=2 or not all(isinstance(x,(int,float)) and math.isfinite(x) for x in interval):return False
            if not 0<=interval[0]<=value<=interval[1]:return False
    return True


def filter_candidates(rows,measurement):
    retained=[];trace=[];uncertain=False
    common=reading_space(rows)['common_row_totals']
    for ri,candidates in enumerate(rows):
        kept=[]
        for ci,c in enumerate(candidates):
            checks=[]
            for role,offset in c['offsets'].items():
                if offset is None:continue
                observed=measurement['rows'][ri]['tiers'][role];x=observed['measured_offset'];lo,hi=observed['offset_interval']
                distance=abs(offset-x);passes=distance<=0.5
                wholly_inside=offset-0.5<=lo and hi<=offset+0.5
                wholly_outside=hi<offset-0.5 or lo>offset+0.5
                boundary_uncertain=not (wholly_inside or wholly_outside)
                checks.append(dict(tier=role,candidate_offset=offset,measured_offset=x,absolute_error=distance,
                                   tolerance=0.5,retained=passes,offset_interval=[lo,hi],boundary_uncertain=boundary_uncertain,definitely_outside=wholly_outside))
            uncertain|=c['row_total'] in common and any(check['boundary_uncertain'] for check in checks) and not any(check['definitely_outside'] for check in checks)
            keep=all(c['retained'] for c in checks)
            trace.append(dict(logical_row_index=ri,human_readable_row_number=ri+1,candidate_index=ci,offsets=c['offsets'],runs=c['runs'],row_total=c['row_total'],retained_by_position=keep,tier_checks=checks))
            if keep:kept.append(c)
        retained.append(kept)
    return retained,trace,uncertain


def decide(r3_decision,decoded_rows,analysis,measurement=None):
    before=r3_decision['outcome'];decision=copy.deepcopy(r3_decision)
    if before not in ('ACCEPT','REVIEW_UNCERTAIN','REJECT_INVALID'):raise ValueError('Unknown R3 outcome')
    result=dict(policy_id=ID,decision=decision,prior_R3_outcome=before,new_review=False,findings=[],
                selected_reading=None,selected_reading_status='not_selected_prior_'+before.lower(),carpet_generation_ready=False,
                distinct_readings_before=None,distinct_readings_after=None,image_position_verified=False,ambiguity_resolved=False,
                candidate_trace=[],measurement_evidence=copy.deepcopy(measurement),image_invalidity_newly_confirmed=False)
    if before!='ACCEPT':return result
    def review(code,message):
        decision['outcome']='REVIEW_UNCERTAIN'
        finding=dict(reason_code=code,stage=ID,explanation=message,recommended_reviewer_action='Verify the row positions and ordered symbol/count runs against the image. No source-image invalidity is confirmed.',
                     affected_rows=sorted({t['logical_row_index'] for t in result['candidate_trace'] if t['retained_by_position'] is False}),
                     affected_tiers=sorted({c['tier'] for t in result['candidate_trace'] for c in t['tier_checks'] if not c['retained']}))
        result['findings'].append(finding)
        decision['reason_codes']=list(dict.fromkeys(decision.get('reason_codes',[])+[code]))
        decision['reason_details']=decision.get('reason_details',[])+[finding]
        # Retain the previous explanation verbatim; new explanation lives here.
        result.update(new_review=True,selected_reading_status='unresolved_'+code)
    try:rows=catalogue(decoded_rows,analysis)
    except (ValueError,KeyError,TypeError) as exc:
        result['candidate_blocker']=str(exc);review('R3_1_ALIGNMENT_UNAVAILABLE',str(exc));return result
    prior=reading_space(rows);result['distinct_readings_before']=prior['distinct_readings']
    if prior['distinct_readings']==0:
        review('R3_1_ALIGNMENT_UNAVAILABLE','No complete canonical page reading with common row totals is available.');return result
    available=measurement_available(measurement,decoded_rows)
    if available:
        remaining,trace,uncertain=filter_candidates(rows,measurement);result['candidate_trace']=trace
        for t in trace:t['eligible_common_row_total_before_filter']=t['row_total'] in prior['common_row_totals']
        if uncertain:
            available=False;result['position_unavailable_reason']='Offset uncertainty straddles the fixed 0.5-column boundary'
        else:
            after=reading_space(remaining);result['distinct_readings_after']=after['distinct_readings'];result['image_position_verified']=True
            for t in trace:
                t['eligible_common_row_total_after_filter']=t['row_total'] in after['common_row_totals']
                t['supports_complete_page_after_filter']=t['retained_by_position'] and t['eligible_common_row_total_after_filter']
            if after['distinct_readings']==1:
                result.update(selected_reading=after['selected'],selected_reading_status='unique_complete_image_supported',carpet_generation_ready=True,
                              ambiguity_resolved=prior['distinct_readings']>1,reading_unique_before_measurement=prior['distinct_readings']==1)
            elif after['distinct_readings']==0:
                review('R3_1_MEASURED_POSITION_MISMATCH','No complete canonical carpet reading matches verified measured positions and common row totals.')
            else:
                review('R3_1_AMBIGUOUS_AFTER_MEASUREMENT','Several distinct complete carpet readings remain after measured-position filtering.')
    if not available:
        result.setdefault('position_unavailable_reason',(measurement or {}).get('reason','Missing or unverified image-position evidence'))
        if not result['candidate_trace']:
            result['candidate_trace']=[dict(logical_row_index=i,human_readable_row_number=i+1,candidate_index=j,offsets=c['offsets'],runs=c['runs'],row_total=c['row_total'],
                                           eligible_common_row_total_before_filter=c['row_total'] in prior['common_row_totals'],retained_by_position=None,
                                           position_check='unavailable',tier_checks=[]) for i,row in enumerate(rows) for j,c in enumerate(row)]
        if prior['distinct_readings']==1:
            result.update(selected_reading=prior['selected'],selected_reading_status='unique_text_only_position_unavailable',reading_unique_before_measurement=True)
        else:
            review('R3_1_POSITION_UNAVAILABLE','Image positioning is unavailable or uncertain, and several distinct canonical carpet readings remain.')
    assert_invariants(before,result)
    return result


def assert_invariants(before,result):
    after=result['decision']['outcome']
    allowed={'ACCEPT':{'ACCEPT','REVIEW_UNCERTAIN'},'REVIEW_UNCERTAIN':{'REVIEW_UNCERTAIN'},'REJECT_INVALID':{'REJECT_INVALID'}}
    if after not in allowed[before]:raise AssertionError('R3.1 monotonicity violation')
    if result['carpet_generation_ready']:
        if after!='ACCEPT' or result['distinct_readings_after']!=1 or not result['image_position_verified'] or not result['selected_reading']:
            raise AssertionError('Unverified or ambiguous carpet generation')
    if after=='ACCEPT' and not result['selected_reading']:raise AssertionError('ACCEPT lacks explicit selected reading')
