"""Monotonic, label-blind R3_HITL_V3. No images, model or mutation inputs."""
from pathlib import Path
import sys,copy
HERE=Path(__file__).resolve().parents[1]
WORKSPACE=Path(r'C:\NAQSHKASH\stage2\stage2f_rejection_evaluation_v2')
sys.path.insert(0,str(WORKSPACE/'src'))
from integration import load_json,digest,content_hash
STAGE1=WORKSPACE/'r3_text_alignment_experiment'
sys.path.insert(0,str(STAGE1/'src'))
from text_alignment import analyze_observation
from talim_validator import validate_aligned_row,validate_tiers
ID='R3_HITL_V3'
OUTCOMES=('ACCEPT','REVIEW_UNCERTAIN','REJECT_INVALID')


def load_policy(path=None):
    c=load_json(path or HERE/'configs/policy_v3.json')
    return validate_policy(c)


def validate_policy(c):
    if (c.get('policy_id')!=ID or c.get('accept_requirement')!='supported_complete_canonical_page' or
        c.get('new_text_action')!='REVIEW_UNCERTAIN' or c.get('preserve_R2_review') is not True or c.get('preserve_R2_reject') is not True or
        type(c.get('max_offset')) is not int or c['max_offset']<0 or type(c.get('max_candidates')) is not int or c['max_candidates']<1 or not isinstance(c.get('dependencies'),dict)):
        raise ValueError('Unsupported R3 policy configuration')
    allowed={'policy_id','accept_requirement','new_text_action','preserve_R2_review','preserve_R2_reject','max_offset','max_candidates','dependencies'}
    if set(c)!=allowed:raise ValueError('Unknown policy fields; no confidence thresholds are supported')
    return c


def verify_dependencies(config):
    for p,h in config['dependencies'].items():
        if digest(p)!=h:raise ValueError('Frozen dependency drift: '+p)


def runtime_inputs(rows,availability,config):
    return dict(decoded_rows=rows,preprocessing_status=availability.get('preprocessing_status','unknown'),
        segmentation_accepted=availability.get('segmentation_accepted',False),ocr_status=availability.get('ocr_status','unknown'),
        max_offset=config['max_offset'],max_candidates=config['max_candidates'])


def evidence(rows,availability,config,*,verified_saved_analysis=None):
    """Saved analysis is admitted only by the evaluator's completed-run verifier."""
    inputs=runtime_inputs(rows,availability,config)
    analysis=analyze_observation(**inputs) if verified_saved_analysis is None else verified_saved_analysis
    return dict(input_sha256=content_hash(inputs),engine_sha256=digest(STAGE1/'src/text_alignment.py'),
        max_offset=config['max_offset'],max_candidates=config['max_candidates'],analysis=copy.deepcopy(analysis))


def _finding(code,index,row,analysis,rows,explanation,action):
    columns=set();failures=[]
    if row is not None and index is not None and index<len(rows):
        text=rows[index]
        for example in row.get('failure_examples',[]):
            offsets=example['offsets'];u=text.get('upper','');l=text.get('lower','')
            if not all(isinstance(s,str) for s in (u,l,text.get('count'))):continue
            check=validate_aligned_row(' '*(offsets.get('upper') or 0)+u if u else '',text['count'],
                ' '*(offsets.get('lower') or 0)+l if l else '')
            col=check.get('evidence',{}).get('column')
            if type(col) is int and code=='R3_COMPLETE_TEXT_CONTRADICTION':columns.add(col)
            failures.append(dict(offsets=offsets,reason_codes=check['reason_codes'],candidate_evidence=check.get('evidence',{})))
    tier=None
    if row and row.get('failure_reasons') and set(row['failure_reasons'])=={'NONCANONICAL_COUNT_TOKEN'}:tier='count'
    if index is not None and index<len(rows):
        syntax=validate_tiers([rows[index]])
        if syntax.get('evidence',{}).get('role') in ('upper','count','lower'):tier=syntax['evidence']['role']
    count_text=rows[index].get('count') if index is not None and index<len(rows) and isinstance(rows[index],dict) else None
    description=('Row '+str(index+1)+(', '+tier+' tier' if tier else '')+': ' if index is not None else '')+explanation
    return dict(reason_code=code,stage=ID,evidence_source='saved_interpreted_OCR_and_frozen_Stage1_text_model',
        logical_row_index=index,human_readable_row_number=index+1 if index is not None else None,tier=tier,
        candidate_text_columns=sorted(columns),human_readable_candidate_columns=[c+1 for c in sorted(columns)],
        candidate_column_scope='reconstructed count-origin text grid; sampled failing candidates; not source pixels',
        candidate_columns_exhaustive=False,observed_token=count_text,observed_token_scope='whole decoded count tier; not a proven token boundary',
        associated_symbol=None,explanation=description,text_model_search_exhaustive=bool(row['coverage']['conditional_text_model_exhaustive']) if row else analysis['all_rows_exhaustive_in_literal_model'],
        image_geometry_verified=False,processed_image_regions=[],source_image_regions=[],
        location_status='candidate_text_columns_only' if columns else 'logical_row_only' if index is not None else 'page_only',
        location_uncertainty='precise source-image invalid glyph location is not verified',
        candidate_offsets=[c['offsets'] for c in row.get('successful_candidates',[])] if row else [],
        candidate_failure_evidence=failures,search_failure_reasons=row.get('failure_reasons',{}) if row else {},
        recommended_reviewer_action=action)


def findings_for(analysis,rows):
    findings=[]
    for i,row in enumerate(analysis['rows']):
        if row.get('conditional_text_contradiction'):
            findings.append(_finding('R3_COMPLETE_TEXT_CONTRADICTION',i,row,analysis,rows,
                'no canonical reconstruction exists in the complete literal text model. This may reflect OCR spacing/character errors; precise image location is unavailable.',
                'Compare this whole row and its count/symbol spacing with the image; do not assume the image is invalid.'))
        elif row['status']=='unsupported':
            f=_finding('R3_UNSUPPORTED_REPRESENTATION',i,row,analysis,rows,'the decoded representation is unsupported by the text-space search.',
                'Check the row transcription and any leading/empty tiers manually.')
            f['unsupported_reason']=row.get('unsupported_reason');findings.append(f)
        elif not row['coverage']['conditional_text_model_exhaustive']:
            findings.append(_finding('R3_SEARCH_CAP_REACHED',i,row,analysis,rows,'the bounded offset search is incomplete; no image-invalidity conclusion follows.',
                'Inspect this row; record unresolved alignment rather than selecting a convenient candidate.'))
        elif row['uncertain_candidates']:
            findings.append(_finding('R3_CANONICAL_INTERPRETATION_UNCERTAIN',i,row,analysis,rows,'canonical interpretation is uncertain for one or more candidates.',
                'Check ambiguous/unsupported glyph interpretations.'))
    if analysis['conditional_text_contradiction'] and not findings:
        f=_finding('R3_ROW_TOTAL_CONFLICT',None,None,analysis,rows,'candidate-derived row totals have no common value; the faulty row is not established.',
            'Compare count transcriptions across all logical rows. Do not force an expected total.')
        f['candidate_logical_rows']=list(range(len(rows)));f['candidate_row_totals']=[r['row_totals'] for r in analysis['rows']];findings.append(f)
        f['human_readable_candidate_rows']=[i+1 for i in f['candidate_logical_rows']]
        f['explanation']='Rows '+', '.join(map(str,f['human_readable_candidate_rows']))+': '+f['explanation']
    if analysis.get('unsupported_reason'):
        findings.append(_finding('R3_OCR_OR_SEGMENTATION_UNAVAILABLE',None,None,analysis,rows,
            'usable preprocessing/OCR evidence is unavailable; a consistent reading cannot be established.','Inspect input readability and segmentation before reviewing transcription.'))
    if not findings and analysis['page_candidate_count_in_search']>1:
        for i,row in enumerate(analysis['rows']):
            usable=[c for c in row['successful_candidates'] if c['row_total'] in analysis['common_row_totals']]
            if len(usable)>1:
                f=_finding('R3_AMBIGUOUS_TEXT_ALIGNMENT',i,row,analysis,rows,
                    'multiple canonical text alignments remain; none establishes the printed symbol/count positions.','Verify the whole row against the image before selecting an interpretation.')
                f['candidate_offsets']=[c['offsets'] for c in usable];findings.append(f)
    if not findings:
        findings.append(_finding('R3_INSUFFICIENT_ALIGNMENT_EVIDENCE',None,None,analysis,rows,
            'an adequately supported complete canonical reading is not established.','Review the transcription and search evidence manually.'))
    return findings


def decide(r2_decision,interpreted_rows,availability,config=None,alignment_evidence=None):
    config=validate_policy(config or load_policy());before=r2_decision['outcome']
    if before not in OUTCOMES:raise ValueError('Unknown R2 outcome')
    ev=alignment_evidence or evidence(interpreted_rows,availability,config)
    if (ev['input_sha256']!=content_hash(runtime_inputs(interpreted_rows,availability,config)) or
        ev['engine_sha256']!=digest(STAGE1/'src/text_alignment.py') or
        ev['max_offset']!=config['max_offset'] or ev['max_candidates']!=config['max_candidates']):raise ValueError('Alignment evidence/source/input mismatch')
    result=copy.deepcopy(r2_decision);analysis=ev['analysis'];findings=[]
    supported=(availability.get('preprocessing_status')=='ok' and availability.get('segmentation_accepted') is True and availability.get('ocr_status')=='ok' and
        analysis.get('status')=='consistent_candidate_found' and
        analysis.get('all_rows_exhaustive_in_literal_model') is True and analysis.get('page_candidate_count_in_search',0)>=1 and not analysis.get('any_validator_uncertainty'))
    if before=='ACCEPT' and not supported:
        findings=findings_for(analysis,interpreted_rows);result['outcome']='REVIEW_UNCERTAIN'
        result['reason_codes']=list(dict.fromkeys(result.get('reason_codes',[])+[f['reason_code'] for f in findings]))
        result['reason_details']=result.get('reason_details',[])+copy.deepcopy(findings)
        result['explanation']='Manual review is required by the R3 text-consistency policy. OCR-only evidence does not prove source-image invalidity.'
    assert_invariants(before,result['outcome'])
    return dict(policy_id=ID,decision=result,findings=findings,alignment_evidence=ev,
        prior_R2_outcome=before,new_review=before=='ACCEPT' and result['outcome']=='REVIEW_UNCERTAIN',
        image_invalidity_newly_confirmed=False)


def assert_invariants(before,after):
    allowed={'ACCEPT':{'ACCEPT','REVIEW_UNCERTAIN'},'REVIEW_UNCERTAIN':{'REVIEW_UNCERTAIN'},'REJECT_INVALID':{'REJECT_INVALID'}}
    if before not in allowed or after not in allowed[before]:raise AssertionError('R3 monotonicity/no-text-reject invariant violated')
