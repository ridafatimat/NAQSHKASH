"""Text-only feasibility, not image validity. No labels, renderer or pixel inputs."""
import sys
from pathlib import Path
from collections import Counter
WORKSPACE=Path(r'C:\NAQSHKASH\stage2\stage2f_rejection_evaluation_v2')
sys.path.insert(0,str(WORKSPACE/'src'))
from integration import rules
from talim_validator import validate_aligned_row

DEFAULT_MAX_OFFSET=128
DEFAULT_MAX_CANDIDATES=258
PREFIX_PROOF=(all(len(s)==1 for s in rules.TRUSTED_SYMBOLS) and
    all(t and not any(c.isspace() for c in t) for t in (rules.encode_count(n) for n in range(1,260))))


def analyze_row(row,max_offset=DEFAULT_MAX_OFFSET,max_candidates=DEFAULT_MAX_CANDIDATES):
    """Enumerate possible strict canonical rows under the prefix-only text model.

    Exact observed characters/spacing are retained. Observed leading blanks are
    unsupported; only missing leading ASCII spaces may be added to otherwise
    leading-stripped strings. No observed spaces are stripped.
    A passing canonical count line starts at zero by canonical_row's contract,
    NOT because an OCR crop establishes its original physical offset.
    """
    if type(max_offset) is not int or max_offset<0 or type(max_candidates) is not int or max_candidates<1:raise ValueError('Invalid computational budget')
    result=dict(status='unsupported',successful_candidates=[],passing_count=0,row_totals=[],
        tested_candidates=0,failure_reasons={},uncertain_candidates=0,failure_examples=[],
        coverage=dict(conditional_text_model_exhaustive=False,image_geometry_exhaustive=False,
            model='literal leading-stripped tier strings plus nonnegative leading prefixes; no insertions/deletions/space repair'))
    def unsupported(reason):result['unsupported_reason']=reason;return result
    if not PREFIX_PROOF:return unsupported('CANONICAL_PREFIX_PROOF_NOT_APPLICABLE')
    if not isinstance(row,dict) or set(row)-{'upper','count','lower'}:return unsupported('UNSUPPORTED_ROW_SCHEMA')
    if 'count' not in row:return unsupported('COUNT_TIER_UNAVAILABLE')
    u,c,l=(row.get(k,'') for k in ('upper','count','lower'))
    if not all(isinstance(t,str) for t in (u,c,l)):return unsupported('NONSTRING_TIER')
    result['observed_edge_spaces']={k:dict(leading=len(t)-len(t.lstrip(' ')),trailing=len(t)-len(t.rstrip(' '))) for k,t in zip(('upper','count','lower'),(u,c,l))}
    if not c:return unsupported('EMPTY_COUNT_OCR_NOT_IMAGE_ABSENCE')
    if any(any(ch.isspace() and ch!=' ' for ch in t) for t in (u,c,l)):return unsupported('NON_ASCII_OR_CONTROL_WHITESPACE')
    if any(t and not t.strip(' ') for t in (u,l)):return unsupported('OPTIONAL_TIER_WHITESPACE_ONLY_IS_NOT_KNOWN_EMPTY')
    if not c.strip(' '):return unsupported('COUNT_WHITESPACE_ONLY')
    if any(t.startswith(' ') for t in (u,c,l) if t):
        return unsupported('OBSERVED_LEADING_BLANKS_REQUIRE_UNPROVEN_COMMON_CROP_NORMALIZATION')
    W=len(c);fits=len(u)<=W and len(l)<=W
    a=W-len(u) if u else 0;b=W-len(l) if l else 0
    first_u=bool(u and u[0]!=' ' and fits);first_l=bool(l and l[0]!=' ' and fits)
    natural=(b+1 if first_u else 0)+(a+1 if first_l else 0)-int(first_u and first_l)
    generated=(min(b,max_offset)+1 if first_u else 0)+(min(a,max_offset)+1 if first_l else 0)-int(first_u and first_l)
    result['coverage'].update(canonical_width=W,natural_symbol_offset_bounds=dict(upper=max(0,a),lower=max(0,b)),
        count_offset_candidates_after_necessary_canonical_pruning=[0],
        canonical_pruning_proof='trusted symbols are length 1; every count encoding is nonempty and blank-free; canonical count starts at 0, width=len(count); first symbol starts at 0',
        natural_candidate_count=natural,generated_with_offset_cap=generated,max_offset=max_offset,max_candidates=max_candidates,
        conditional_text_model_exhaustive=generated==natural and generated<=max_candidates)
    reasons=Counter();uncertain=0
    def pairs():
        if first_u:
            for ol in range(min(b,max_offset)+1):yield 0,ol
        if first_l:
            for ou in range(min(a,max_offset)+1):
                if ou==0 and first_u:continue
                yield ou,0
    for ou,ol in pairs():
        if result['tested_candidates']>=max_candidates:break
        upper=' '*ou+u if u else '';lower=' '*ol+l if l else ''
        # Crucially, no expected_cells from labels or source metadata.
        validation=validate_aligned_row(upper,c,lower)
        result['tested_candidates']+=1
        offsets=dict(upper=ou if u else None,count=0,lower=ol if l else None)
        if validation['outcome']=='ACCEPT':
            runs=validation['evidence']['runs'];total=sum(r['count'] for r in runs)
            result['successful_candidates'].append(dict(offsets=offsets,row_total=total,runs=runs))
        else:
            uncertain+=validation['outcome']=='REVIEW_UNCERTAIN'
            codes=validation['reason_codes'];reasons.update(codes)
            if len(result['failure_examples'])<6:result['failure_examples'].append(dict(offsets=offsets,outcome=validation['outcome'],reasons=codes))
    if natural==0:reasons['NO_CANONICAL_ORIGIN_OR_TIER_EXCEEDS_COUNT_WIDTH']+=1
    result.update(failure_reasons=dict(reasons),uncertain_candidates=uncertain,
        passing_count=len(result['successful_candidates']),row_totals=sorted({x['row_total'] for x in result['successful_candidates']}))
    result['status']='multiple_in_search' if result['passing_count']>1 else 'one_in_search' if result['passing_count'] else 'none_in_search'
    result['conditional_text_contradiction']=not result['passing_count'] and not uncertain and result['coverage']['conditional_text_model_exhaustive']
    result['unique_reconstruction_proven_in_model']=result['passing_count']==1 and not uncertain and result['coverage']['conditional_text_model_exhaustive']
    return result


def combine_rows(rows):
    totals=[set(r['row_totals']) for r in rows]
    common=set.intersection(*totals) if totals else set()
    exhaustive=bool(rows and all(r['status']!='unsupported' and r['coverage']['conditional_text_model_exhaustive'] and not r['uncertain_candidates'] for r in rows))
    contradicted=any(r.get('conditional_text_contradiction') for r in rows) or bool(exhaustive and not common)
    count=0
    for total in common:
        combinations=1
        for row in rows:combinations*=sum(c['row_total']==total for c in row['successful_candidates'])
        count+=combinations
    return dict(status='consistent_candidate_found' if common else 'text_contradiction_complete_model' if contradicted else 'unknown_incomplete_or_unsupported',
        rows=rows,common_row_totals=sorted(common),multiple_common_totals=len(common)>1,
        page_candidate_count_in_search=count,page_multiplicity='multiple_in_search' if count>1 else 'one_in_search' if count==1 else 'none_in_search',
        unique_page_alignment_proven_in_model=count==1 and exhaustive,
        totals_inferred_only_from_candidates=True,all_rows_exhaustive_in_literal_model=exhaustive,
        image_geometry_exhaustive=False,conditional_text_contradiction=contradicted,
        all_rows_have_passing_candidates=bool(rows and all(r['passing_count'] for r in rows)),
        any_multiple_candidates=any(r['passing_count']>1 for r in rows),
        any_validator_uncertainty=any(r['uncertain_candidates'] for r in rows),
        some_row_has_no_passing_candidate=any(r['status']=='none_in_search' for r in rows),
        all_candidates_fail_in_tested_search=bool(rows and all(r['status']!='unsupported' for r in rows) and not count),
        search_scope='text reconstruction only; never proves source-image validity or invalidity')


def analyze_observation(decoded_rows,*,preprocessing_status='ok',segmentation_accepted=True,ocr_status='ok',
        max_offset=DEFAULT_MAX_OFFSET,max_candidates=DEFAULT_MAX_CANDIDATES):
    if preprocessing_status!='ok' or not segmentation_accepted or ocr_status!='ok' or not isinstance(decoded_rows,list) or not decoded_rows:
        result=combine_rows([]);result['unsupported_reason']='D1_OR_OCR_UNAVAILABLE';return result
    return combine_rows([analyze_row(r,max_offset,max_candidates) for r in decoded_rows])


def review_only_comparison(original_outcome,analysis):
    """A diagnostic counterfactual, not an installed Talim decision policy."""
    return 'REVIEW_UNCERTAIN' if original_outcome=='ACCEPT' and analysis['conditional_text_contradiction'] else original_outcome
