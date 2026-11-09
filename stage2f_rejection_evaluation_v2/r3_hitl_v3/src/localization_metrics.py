"""OFFLINE only: mutation labels enter after policy and region planning."""
from collections import Counter


def score(record,source):
    """Eligible conditional metrics stay distinct from broad-region recall."""
    findings=record['findings'];regions=record['review_regions']['regions']
    availability=dict(explanation_available=bool(findings or record['R3'].get('explanation')),
        logical_row_available=any(type(f['logical_row_index']) is int for f in findings),
        candidate_text_columns_available=any(f['candidate_text_columns'] for f in findings),
        processed_region_available=bool(regions),source_region_available=bool(regions))
    result=dict(availability=availability,scope='new_invalid_review_only',metrics={},precise_pixel_localization_eligible=False)
    if source['validity_status']=='VALID':result['scope']='valid_review_no_true_invalid_mutation';return result
    if not record['new_review']:result['scope']='not_a_new_R3_review';return result
    p=source['mutation_parameters'];row=p['row'];tier='count' if p['target_tier']=='count_line' else p['target_tier']
    row_map=record['OCR_available'] and len(record['interpreted_transcription'])==len(source['raw_display'])
    rows={f['logical_row_index'] for f in findings if type(f['logical_row_index']) is int}
    tiers={f['tier'] for f in findings if f['logical_row_index']==row and f['tier'] in ('upper','count','lower')}
    cols={c for f in findings if f['logical_row_index']==row for c in f['candidate_text_columns']}
    reference=source['raw_display'][row]['count_line'];pred=record['interpreted_transcription'][row].get('count') if row<len(record['interpreted_transcription']) else None
    comparable_columns=bool(row_map and reference and reference[0]!=' ' and pred==reference.rstrip(' '))
    m=dict(logical_row=dict(eligible=bool(row_map and rows),candidate_hit=row in rows,unique_correct=rows=={row},extra_rows=sorted(rows-{row})),
        tier=dict(eligible=bool(row_map and tiers),candidate_hit=tier in tiers,unique_correct=tiers=={tier}),
        text_column=dict(eligible=bool(comparable_columns and cols),candidate_hit=p['column'] in cols,
            scope='sampled candidate-grid membership only; not precise physical localization',
            unscorable_reason=None if comparable_columns and cols else 'No supported comparable count-origin text columns'))
    # Source mutation CELL rectangles, not a claimed exact changed-ink oracle.
    layout=source['layout'];x=layout['padding']+layout['extra_left']+p['column']*layout['cell_width']
    target_roles=[p['target_tier']]+([p['source_tier']] if p['source_tier']!=p['target_tier'] else [])
    boxes=[]
    for role in target_roles:
        y=layout['padding']+row*(3*layout['line_height']+layout['row_gap'])+{'upper':0,'count_line':1,'lower':2}[role]*layout['line_height']
        boxes.append([x,y,x+p['width']*layout['cell_width'],y+layout['line_height']])
    def overlaps(a,b):return max(a[0],b[0])<min(a[2],b[2]) and max(a[1],b[1])<min(a[3],b[3])
    good=[r for r in regions if r.get('image_geometry_verified')]
    m['broad_image_region']=dict(eligible=bool(good),candidate_hit=any(overlaps(r['source_box'],b) for r in good for b in boxes),
        scope='broad row/tier overlap with source mutation cell(s); NOT precise mutation localization')
    result.update(metrics=m,location_unknown=not any(z['eligible'] for z in m.values()),
        any_incorrect_eligible=any(z['eligible'] and not z['candidate_hit'] for z in m.values()))
    return result


def summarize(records):
    targeted=[r for r in records if r['localization']['scope']=='new_invalid_review_only'];denom=len(targeted)
    result=dict(new_invalid_reviews=denom,source_parents=len({r['source_parent_id'] for r in targeted}),metrics={})
    def ratio(a,b):return a/b if b else None
    for key in ('logical_row','tier','text_column','broad_image_region'):
        eligible=[r for r in targeted if r['localization']['metrics'].get(key,{}).get('eligible')]
        hits=[r for r in eligible if r['localization']['metrics'][key]['candidate_hit']]
        result['metrics'][key]=dict(eligible=len(eligible),correct_candidate_hits=len(hits),incorrect=len(eligible)-len(hits),
            coverage=ratio(len(eligible),denom),conditional_candidate_accuracy=ratio(len(hits),len(eligible)),
            full_denominator_candidate_recall=ratio(len(hits),denom),eligible_parents=len({r['source_parent_id'] for r in eligible}))
        if key in ('logical_row','tier'):
            result['metrics'][key]['unique_correct']=sum(r['localization']['metrics'][key]['unique_correct'] for r in eligible)
    result['precise_pixel_localization']=dict(eligible=0,coverage=0 if denom else None,conditional_accuracy=None)
    result['location_unknown']=sum(r['localization'].get('location_unknown',True) for r in targeted)
    result['any_incorrect_eligible']=sum(r['localization'].get('any_incorrect_eligible',False) for r in targeted)
    valid=[r for r in records if r['validity']=='VALID' and r['R3']['outcome']=='REVIEW_UNCERTAIN']
    result['valid_review_availability']=dict(images=len(valid),**{k:sum(r['localization']['availability'][k] for r in valid) for k in
        ('explanation_available','logical_row_available','candidate_text_columns_available','processed_region_available','source_region_available')},
        true_invalid_mutation_location='NOT_APPLICABLE')
    return result
