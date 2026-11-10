"""Source notation/labels enter ONLY after the complete R3.1 decision."""
from collections import Counter


def ratio(a,b):return a/b if b else None


def score(result,measurement,source,decoded_rows):
    reference=source['raw_display'];offsets=[]
    measured=(measurement or {}).get('rows',[])
    row_mapping=len(reference)==len(decoded_rows)==len(measured)
    for i,r in enumerate(measured):
        for role,t in r['tiers'].items():
            item=dict(row=i,tier=role,first_ink_measured=t.get('first_ink_x') is not None,offset_eligible=False)
            if row_mapping:
                truth={('count' if k=='count_line' else k):v for k,v in reference[i].items() if v.strip(' ')}
                starts={k:len(v)-len(v.lstrip(' ')) for k,v in truth.items()}
                if set(truth)==set(r['tiers']) and role in starts and 'measured_offset' in t:
                    expected=starts[role]-min(starts.values());error=t['measured_offset']-expected
                    item.update(offset_eligible=True,expected_offset=expected,measured_offset=t['measured_offset'],error_columns=error,
                                exact=abs(error)<=1e-9,within_half_column=abs(error)<=0.5)
            if not item['offset_eligible']:item['unscorable_reason']='No verified complete pitch/row/tier correspondence to MUTATED notation'
            offsets.append(item)
    valid=source['validity_status']=='VALID';selected=result['selected_reading']
    selected_eligible=bool(valid and result['decision']['outcome']=='ACCEPT' and selected)
    expected=[r['runs'] for r in source['original_structured_notation']['logical_rows']] if selected_eligible else None
    actual=[r['runs'] for r in selected['logical_rows']] if selected_eligible else None
    return dict(tiers=offsets,selected_reading_eligible=selected_eligible,selected_reading_correct=actual==expected if selected_eligible else None,
                selected_reading_comparison='exact ordered symbol/count runs; no glyph-alias normalization',
                invalid_selected_reading_truth='UNSCORABLE: genuinely invalid image has no canonical reference carpet reading' if not valid else None,
                source_row_mapping_eligible=row_mapping)


def summary(records):
    tiers=[t for r in records for t in r['offline_score']['tiers']];eligible=[t for t in tiers if t['offset_eligible']]
    selected=[r for r in records if r['offline_score']['selected_reading_eligible']]
    correct=sum(r['offline_score']['selected_reading_correct'] for r in selected)
    valid_accepts=sum(r['validity']=='VALID' and r['R3_1']['decision']['outcome']=='ACCEPT' for r in records)
    return dict(measured_tiers=len(tiers),first_ink_measured=sum(t['first_ink_measured'] for t in tiers),offset_eligible=len(eligible),
                exact_offsets=sum(t['exact'] for t in eligible),within_half_column=sum(t['within_half_column'] for t in eligible),
                conditional_exact_offset_rate=ratio(sum(t['exact'] for t in eligible),len(eligible)),
                offset_errors_columns=[t['error_columns'] for t in eligible],
                valid_ACCEPT_denominator=valid_accepts,selected_reading_eligible=len(selected),selected_reading_correct=correct,
                selected_reading_false=len(selected)-correct,correct_selected_reading_rate=ratio(correct,len(selected)),
                false_selected_reading_rate=ratio(len(selected)-correct,len(selected)),selected_reading_coverage=ratio(len(selected),valid_accepts),
                ready_valid_selected_reading_false=sum(r['R3_1']['carpet_generation_ready'] and not r['offline_score']['selected_reading_correct'] for r in selected),
                invalid_ACCEPT_selected_readings_unscorable=sum(r['validity']=='INVALID' and r['R3_1']['selected_reading'] is not None for r in records),
                unscorable_offset_reasons=dict(Counter(t['unscorable_reason'] for t in tiers if not t['offset_eligible'])))
