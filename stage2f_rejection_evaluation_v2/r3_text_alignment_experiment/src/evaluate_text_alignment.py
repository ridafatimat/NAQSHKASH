"""MANUAL Stage 1 evaluation of saved development text; no OCR or image reads."""
import argparse,json,sys
from pathlib import Path
from collections import Counter
HERE=Path(__file__).resolve().parents[1]
WORKSPACE=Path(r'C:\NAQSHKASH\stage2\stage2f_rejection_evaluation_v2')
sys.path.insert(0,str(WORKSPACE/'src'))
from integration import load_json,digest,content_hash,write_json,write_csv,CHECKPOINT_SHA
from dataset_loader import DevelopmentDataset,SPLIT
from evaluate_r2 import verify_r2
from observation_store import load_raw_run
from r3_fidelity_diagnostic import normalized
from measurements import alignment_targets
sys.path.insert(0,str(HERE/'src'))
from text_alignment import analyze_observation,review_only_comparison,DEFAULT_MAX_OFFSET,DEFAULT_MAX_CANDIDATES
FONT_PROFILE=load_json(WORKSPACE.parent/'stage2f_tier_aware_decoding/reports/glyph_audit_summary.json')


def load_inputs(r2_dir=None):
    verify_r2();dataset=DevelopmentDataset(verify_files=False)
    if r2_dir is None:
        matches=[p.parent for p in sorted((WORKSPACE/'metrics').glob('*/results.json')) if load_json(p).get('candidate')=='R2']
        if len(matches)!=1:raise ValueError('Select one completed R2 run explicitly with --r2-dir')
        r2_dir=matches[0]
    r2_dir=Path(r2_dir).resolve()
    if not r2_dir.is_relative_to((WORKSPACE/'metrics').resolve()):raise ValueError('Use an existing metrics R2 run')
    result=load_json(r2_dir/'results.json');done=load_json(r2_dir/'completion.json');lineage=result['lineage']
    if (result['candidate']!='R2' or done['candidate']!='R2' or not done['complete'] or done['records']!=2800 or
        done['results_sha256']!=digest(r2_dir/'results.json') or result['r2_lock_sha256']!=digest(WORKSPACE/'configs/r2_lock.json') or
        lineage['dataset_manifest_sha256']!=dataset.manifest_sha256 or lineage['selection_sha256']!=dataset.selection_sha256 or
        lineage['checkpoint_sha256']!=CHECKPOINT_SHA or lineage['experiment_lock_sha256']!=digest(WORKSPACE/'configs/experiment_lock.json')):
        raise ValueError('R2 completion/lineage differs')
    rows=result['rows']
    if len(rows)!=2800 or {r['image_id'] for r in rows}!=set(dataset.by_id):raise ValueError('R2 must contain exactly development coverage')
    raw_paths=[p for p in (WORKSPACE/'metrics').glob('*/observations.jsonl') if digest(p)==lineage['raw_observations_sha256']]
    if len(raw_paths)!=1:raise ValueError('Cannot uniquely locate shared raw observations')
    raw,raw_lineage=load_raw_run(raw_paths[0].parent,dataset)
    if raw_lineage['completion']['plan_sha256']!=lineage['raw_plan_sha256']:raise ValueError('Raw plan differs')
    raw={x['identity']['image_id']:x['observation'] for x in raw}
    for r in rows:
        source=dataset.by_id[r['image_id']]
        for a,b in [('source_parent_id','source_parent_id'),('split','dataset_split'),('image_sha256','image_sha256'),('validity','validity_status'),('invalidity_category','invalidity_category')]:
            if r[a]!=source[b]:raise ValueError('R2 identity/evaluation-label drift')
        if r['split']!=SPLIT or r['raw_OCR_transcription']!=raw[r['image_id']]['decoded_rows']:raise ValueError('R2/raw observation mismatch')
    return dataset,rows,raw,dict(r2_dir=str(r2_dir),r2_results_sha256=digest(r2_dir/'results.json'),
        raw_dir=str(raw_paths[0].parent),raw_sha256=digest(raw_paths[0]),dataset_manifest_sha256=dataset.manifest_sha256)


def offline_cohorts(source,r2):
    """Labels/layout/mutations restricted to post-decision scoring."""
    verified=source['layout']['font_size_px']==48 and source['layout']['font_sha256']==FONT_PROFILE['font_sha256']
    predicted=r2['interpreted_transcription'];reference=source['raw_display']
    pattern=(normalized(predicted,aliases=True)==normalized(reference,aliases=True)) if verified else None
    glyphs=(normalized(predicted,ink=True,aliases=True)==normalized(reference,ink=True,aliases=True)) if verified else None
    preserved_reversal=False;gap_loss=False
    if source['validity_status']=='INVALID':
        p=source['mutation_parameters'];ri=p['row']
        actual=predicted[ri].get('count','') if ri<len(predicted) else None
        text=reference[ri]['count_line'].strip(' ')
        gap_loss=actual is not None and ' ' in text and actual!=text and actual==text.replace(' ','')
        if source['invalidity_category']=='invalid_count_token' and actual is not None:
            ref=reference[ri]['count_line'].replace(' ','');pred=actual.replace(' ','')
            start=len(reference[ri]['count_line'][:p['column']].replace(' ',''));token=p['displayed_mutated_token']
            preserved_reversal=ref==pred or all(alignment_targets(ref,pred,i)=={ref[i]} for i in range(start,start+len(token)))
    invalid_accept=source['validity_status']=='INVALID' and r2['actual_outcome']=='ACCEPT'
    return dict(pattern_faithful=pattern,glyph_faithful=glyphs,reference_scope_verified=verified,
        invalid_R2_accept=invalid_accept,pattern_faithful_invalid_R2_accept=bool(pattern and invalid_accept),
        preserved_reversal_invalid_R2_accept=bool(preserved_reversal and invalid_accept),
        lost_internal_count_gap=gap_loss,wrong_tier_substituted_invalid_R2_accept=invalid_accept and source['invalidity_category']=='wrong_tier')


def evaluate_case(r2,raw,source,max_offset,max_candidates,cache=None):
    # Runtime API receives only predicted text, observed availability and budgets.
    inputs=dict(decoded_rows=r2['interpreted_transcription'],preprocessing_status=raw['preprocessing_status'],
        segmentation_accepted=raw['segmentation_accepted'],ocr_status=raw['ocr_status'],max_offset=max_offset,max_candidates=max_candidates)
    key=content_hash(inputs)
    if cache is not None and key in cache:analysis=cache[key]
    else:
        analysis=analyze_observation(**inputs)
        if cache is not None:cache[key]=analysis
    counterfactual=review_only_comparison(r2['actual_outcome'],analysis)
    cohorts=offline_cohorts(source,r2)  # only AFTER text inference/counterfactual
    return dict(image_id=r2['image_id'],parent=r2['source_parent_id'],validity=source['validity_status'],
        category=source['invalidity_category'],variant='illuminated' if source['augmentation_recipe'] else 'clean',
        original_R2_outcome=r2['actual_outcome'],review_only_counterfactual=counterfactual,
        OCR_available=raw['preprocessing_status']=='ok' and raw['segmentation_accepted'] and raw['ocr_status']=='ok',
        analysis=analysis,offline_scoring=cohorts)


def summarize(cases):
    def count(predicate):
        selected=[r for r in cases if predicate(r)]
        return dict(images=len(selected),parents=len({r['parent'] for r in selected}))
    flags=dict(plausible_alignment=lambda r:r['analysis']['status']=='consistent_candidate_found',
        complete_model_contradiction=lambda r:r['analysis']['conditional_text_contradiction'],
        all_candidates_fail_in_tested_search=lambda r:r['analysis']['all_candidates_fail_in_tested_search'],
        incomplete_or_unscorable=lambda r:not r['analysis']['all_rows_exhaustive_in_literal_model'],
        multiple_candidates=lambda r:r['analysis']['any_multiple_candidates'],
        one_page_candidate_in_search=lambda r:r['analysis']['page_candidate_count_in_search']==1,
        multiple_page_candidates_in_search=lambda r:r['analysis']['page_candidate_count_in_search']>1,
        unique_page_reconstruction_proven_in_model=lambda r:r['analysis']['unique_page_alignment_proven_in_model'],
        multiple_common_totals=lambda r:r['analysis']['multiple_common_totals'],
        ACCEPT_to_REVIEW=lambda r:r['original_R2_outcome']=='ACCEPT' and r['review_only_counterfactual']=='REVIEW_UNCERTAIN',
        invalid_ACCEPT_to_REVIEW=lambda r:r['validity']=='INVALID' and r['original_R2_outcome']=='ACCEPT' and r['review_only_counterfactual']=='REVIEW_UNCERTAIN',
        valid_ACCEPT_to_REVIEW=lambda r:r['validity']=='VALID' and r['original_R2_outcome']=='ACCEPT' and r['review_only_counterfactual']=='REVIEW_UNCERTAIN')
    return dict(images=len(cases),parents=len({r['parent'] for r in cases}),**{k:count(p) for k,p in flags.items()},
        R2_confusion={v:dict(Counter(r['original_R2_outcome'] for r in cases if r['validity']==v)) for v in ('VALID','INVALID')},
        review_only_confusion={v:dict(Counter(r['review_only_counterfactual'] for r in cases if r['validity']==v)) for v in ('VALID','INVALID')},
        passing_row_multiplicity=dict(Counter(row['status'] for r in cases for row in r['analysis']['rows'])),
        no_new_definitive_rejection=all(r['review_only_counterfactual']!='REJECT_INVALID' or r['original_R2_outcome']=='REJECT_INVALID' for r in cases))


def evaluate_records(dataset,rows,raw,max_offset,max_candidates):
    cache={};cases=[evaluate_case(r,raw[r['image_id']],dataset.by_id[r['image_id']],max_offset,max_candidates,cache) for r in rows]
    result=dict(overall=summarize(cases))
    for field in ('validity','category','variant','parent','OCR_available'):
        result['by_'+field]={str(k):summarize([r for r in cases if r[field]==k]) for k in sorted({r[field] for r in cases})}
    result['by_fidelity']={k:summarize([r for r in cases if r['offline_scoring']['pattern_faithful'] is value]) for k,value in [('faithful',True),('unfaithful',False),('unverified',None)]}
    result['by_glyph_fidelity']={k:summarize([r for r in cases if r['offline_scoring']['glyph_faithful'] is value]) for k,value in [('faithful',True),('unfaithful',False),('unverified',None)]}
    result['targeted_cohorts']={k:summarize([r for r in cases if r['offline_scoring'][k]]) for k in
        ('invalid_R2_accept','pattern_faithful_invalid_R2_accept','preserved_reversal_invalid_R2_accept','lost_internal_count_gap','wrong_tier_substituted_invalid_R2_accept')}
    result['computed_distinct_text_pages']=len(cache)
    return cases,result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--r2-dir',type=Path);p.add_argument('--preflight',action='store_true')
    p.add_argument('--output-dir',type=Path);p.add_argument('--max-offset',type=int,default=DEFAULT_MAX_OFFSET);p.add_argument('--max-candidates',type=int,default=DEFAULT_MAX_CANDIDATES)
    args=p.parse_args()
    if args.max_offset<0 or args.max_candidates<1:p.error('Invalid computational bounds')
    if not args.preflight and args.output_dir is None:p.error('--output-dir is required for evaluation')
    dataset,rows,raw,provenance=load_inputs(args.r2_dir)
    if args.preflight:
        print(json.dumps(dict(images=len(rows),parents=len(dataset.groups),splits=[SPLIT],source=provenance,alignment_evaluation_executed=False),indent=2));return
    out=args.output_dir.resolve()
    if not out.is_relative_to((HERE/'runs').resolve()):raise ValueError('Output must be a new directory below this experiment/runs')
    if out.exists():raise FileExistsError('Do not overwrite evaluation output')
    cases,result=evaluate_records(dataset,rows,raw,args.max_offset,args.max_candidates)
    out.mkdir(parents=True,exist_ok=False)
    provenance.update(max_offset=args.max_offset,max_candidates=args.max_candidates,
        source_text='R2 saved T1 interpreted tiers; all rows; no OCR/space correction',
        offline_projection_sha256=digest(WORKSPACE/'src/r3_fidelity_diagnostic.py'),
        code_sha256={f:digest(HERE/'src'/f) for f in ('text_alignment.py','evaluate_text_alignment.py')})
    write_json(out/'metrics.json',dict(provenance=provenance,**result))
    with (out/'per_image.jsonl').open('x',encoding='utf-8') as f:
        for r in cases:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
    write_csv(out/'comparison.csv',[dict(image_id=r['image_id'],parent=r['parent'],validity=r['validity'],category=r['category'],variant=r['variant'],
        R2=r['original_R2_outcome'],review_only=r['review_only_counterfactual'],status=r['analysis']['status'],
        exhaustive_in_literal_model=r['analysis']['all_rows_exhaustive_in_literal_model'],common_totals=r['analysis']['common_row_totals'],
        multiple_candidates=r['analysis']['any_multiple_candidates'],**r['offline_scoring']) for r in cases])
    text=('# R3 Stage 1 — measured development text-alignment results\n\n'+
        'Conditional literal-text contradictions are not image invalidity. The counterfactual changes ACCEPT to REVIEW only, preserves existing REVIEW/REJECT and never modifies R2. Incomplete searches cannot establish contradictions.\n\n')
    text+='~~~json\n'+json.dumps(result['overall'],indent=2)+'\n~~~\n\nTargeted cohorts (measured, not forced to a hypothesis):\n\n~~~json\n'+json.dumps(result['targeted_cohorts'],indent=2)+'\n~~~\n\n'
    text+='See metrics.json for validity/category/variant/parent/OCR/fidelity breakdowns and per_image.jsonl for row candidates, inferred totals, coverage and failure reasons. Parent counts within conditions overlap and illumination variants are not independent.\n\n'
    text+='Text-space alignment can find a strict canonical reconstruction or disprove all reconstructions under its literal prefix-only model. It cannot recover source geometry, repaired spaces, lost glyphs, true count absence or wrong glyph identities. A passing candidate is a plausible alternative text interpretation, not source validity. Review how many valid controls acquire contradictions and how much ambiguity/coverage remains before designing Stage 2. Use CTC for observed character/run positions and image components for independent glyph/absence support; the measured report must inform that choice. No production rejection policy is selected.\n'
    (out/'report.md').write_text(text,encoding='utf-8')
    write_json(out/'completion.json',dict(complete=True,images=len(cases),metrics_sha256=digest(out/'metrics.json'),evidence_sha256=digest(out/'per_image.jsonl')))
    verify_r2();print(out/'report.md')


if __name__=='__main__':main()
