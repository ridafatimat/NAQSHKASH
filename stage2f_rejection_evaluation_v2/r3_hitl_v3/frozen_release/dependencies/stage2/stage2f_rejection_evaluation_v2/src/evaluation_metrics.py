"""Three-way semantic outcomes and valid-reference transcription metrics only."""
from collections import Counter,defaultdict
from integration import *

OUTCOMES=('ACCEPT','REJECT_INVALID','REVIEW_UNCERTAIN')


def ratio(n,d):return n/d if d else None


def summarize(rows):
    valid=[r for r in rows if r['validity']=='VALID'];invalid=[r for r in rows if r['validity']=='INVALID']
    scorable=[r for r in valid if r['interpreted_errors'] is not None and r['interpreted_errors']['characters']]
    accepted=[r for r in scorable if r['actual_outcome']=='ACCEPT']
    def count(g,outcome):return sum(r['actual_outcome']==outcome for r in g)
    confusion={status:{outcome:sum(r['validity']==status and r['actual_outcome']==outcome for r in rows) for outcome in OUTCOMES} for status in ('VALID','INVALID')}
    corrections=[e['assessment'] for r in rows for e in r['correction_assessments']]
    denom=sum(r['interpreted_errors']['characters'] for r in scorable)
    reasons=Counter(code for r in rows for code in set(r['triggered_reason_codes']))
    return dict(images=len(rows),valid_denominator=len(valid),invalid_denominator=len(invalid),
        three_way_confusion=confusion,true_invalid_rejection_rate=ratio(count(invalid,'REJECT_INVALID'),len(invalid)),
        invalid_accepted_rate=ratio(count(invalid,'ACCEPT'),len(invalid)),invalid_review_rate=ratio(count(invalid,'REVIEW_UNCERTAIN'),len(invalid)),
        invalid_images_incorrectly_accepted_as_valid=count(invalid,'ACCEPT'),
        invalid_interpreted_syntax_passes=sum(r['interpreted_syntax']['outcome']=='ACCEPT' for r in invalid),
        invalid_interpreted_syntax_pass_rate=ratio(sum(r['interpreted_syntax']['outcome']=='ACCEPT' for r in invalid),len(invalid)),
        valid_acceptance_rate=ratio(count(valid,'ACCEPT'),len(valid)),false_definitive_rejection_rate=ratio(count(valid,'REJECT_INVALID'),len(valid)),
        valid_review_rate=ratio(count(valid,'REVIEW_UNCERTAIN'),len(valid)),
        review_rate=ratio(count(rows,'REVIEW_UNCERTAIN'),len(rows)),
        exact_transcription_accuracy=ratio(sum(r['interpreted_errors']['exact'] for r in scorable),len(scorable)),
        raw_exact_transcription_accuracy=ratio(sum(r['raw_errors']['exact'] for r in scorable),len(scorable)),
        valid_raw_CER=ratio(sum(r['raw_errors']['edits'] for r in scorable),denom),
        valid_interpreted_CER=ratio(sum(r['interpreted_errors']['edits'] for r in scorable),denom),
        valid_transcription_denominator=len(scorable),valid_character_denominator=denom,
        accepted_output_transcription_correctness=ratio(sum(r['interpreted_errors']['exact'] for r in accepted),len(accepted)),
        accepted_transcription_denominator=len(accepted),
        valid_end_to_end_page_correctness=ratio(sum(r['actual_outcome']=='ACCEPT' and r['interpreted_errors']['exact'] for r in scorable),len(scorable)),
        semantic_expected_outcome_accuracy=ratio(sum(r['expected_outcome']==r['actual_outcome'] for r in rows),len(rows)),
        preprocessing_failures=sum(r['preprocessing_failure'] for r in rows),
        D1_segmentation_failures=sum(r['segmentation_failure'] for r in rows),OCR_decoding_failures=sum(r['ocr_failure'] for r in rows),
        pipeline_exceptions=sum(r['pipeline_exception'] is not None for r in rows),
        raw_wrong_tier_images=sum('DECODED_UNSUPPORTED_OR_WRONG_TIER_GLYPH' in r['raw_decision']['reason_codes'] for r in rows),
        wrong_tier_reviews=sum(r['actual_outcome']=='REVIEW_UNCERTAIN' and 'DECODED_UNSUPPORTED_OR_WRONG_TIER_GLYPH' in r['triggered_reason_codes'] for r in rows),
        corrected_images=sum(bool(r['corrections']) for r in rows),contextual_edits=len(corrections),
        correct_T1_edits=corrections.count('correct'),incorrect_T1_edits=corrections.count('incorrect'),
        unknown_or_unscorable_T1_edits=sum(s not in ('correct','incorrect') for s in corrections),
        valid_images_with_worsened_edits=sum(r['interpreted_errors']['edits']>r['raw_errors']['edits'] for r in scorable),
        rule_trigger_image_counts=dict(reasons))


def worst_case(rows):
    invalid_accept=any(r['validity']=='INVALID' and r['actual_outcome']=='ACCEPT' for r in rows)
    valid_reject=any(r['validity']=='VALID' and r['actual_outcome']=='REJECT_INVALID' for r in rows)
    accepted_wrong=any(r['validity']=='VALID' and r['actual_outcome']=='ACCEPT' and r['interpreted_errors'] is not None and not r['interpreted_errors']['exact'] for r in rows)
    review=any(r['actual_outcome']=='REVIEW_UNCERTAIN' for r in rows)
    return dict(any_invalid_accepted=invalid_accept,any_valid_definitively_rejected=valid_reject,
                any_accepted_valid_transcription_wrong=accepted_wrong,any_review=review,
                worst_case_level='critical_failure' if invalid_accept or valid_reject or accepted_wrong else 'unresolved_review' if review else 'all_expected_correct')


def grouped(rows):
    parents={pid:dict(metrics=summarize([r for r in rows if r['source_parent_id']==pid]),
                      worst_case=worst_case([r for r in rows if r['source_parent_id']==pid])) for pid in sorted({r['source_parent_id'] for r in rows})}
    groups={}
    for field in ('invalidity_category','paired_mutation_category','variant','length_group','row_group'):
        groups[field]={key:summarize([r for r in rows if r[field]==key]) for key in sorted({r[field] for r in rows})}
    result=summarize(rows);result['source_parents']=len(parents)
    result['macro_parent']={key:ratio(sum(values),len(values)) for key in ('true_invalid_rejection_rate','invalid_accepted_rate','valid_acceptance_rate',
        'false_definitive_rejection_rate','invalid_review_rate','valid_review_rate','exact_transcription_accuracy','accepted_output_transcription_correctness','valid_interpreted_CER')
        for values in [[p['metrics'][key] for p in parents.values() if p['metrics'][key] is not None]]}
    result['parent_worst_case_counts']=dict(Counter(p['worst_case']['worst_case_level'] for p in parents.values()))
    return result,parents,groups
