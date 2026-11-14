"""Manual R0/R1 decisions on exactly one shared completed development OCR run."""
import argparse
from integration import *
from dataset_loader import DevelopmentDataset,SPLIT,expected_valid_rows
from observation_store import load_raw_run
from evaluation_metrics import grouped


def evaluate_case(raw,row,candidate,paired_category):
    # No dataset row/truth enters the runtime decision API.
    result=apply_candidate(raw['observation'],raw['rendering_context'],candidate)
    original=result['raw_observation']['decoded_rows'];interpreted=result['interpreted_observation']['decoded_rows']
    decision=result['final_decision'];truth=expected_valid_rows(row)
    corrections=result['corrections']
    assessments=(assess_corrections(truth,original,corrections) if truth is not None else
                 [dict(e,assessment='unscorable_invalid_no_CTC_reference') for e in corrections])
    max_length=max((len(s.lstrip(' ')) for r in row['raw_display'] for s in r.values()),default=0)
    obs=raw['observation'];exception=obs.get('pipeline_exception')
    preprocessing_failure=obs['preprocessing_status'] in ('unreadable','no_ink') or str(obs.get('segmentation_reason','')).startswith('preprocessing:')
    segmentation_failure=obs['preprocessing_status']=='ok' and not obs['segmentation_accepted'] and not preprocessing_failure
    ocr_failure=obs['segmentation_accepted'] and obs['ocr_status']!='ok'
    return dict(image_id=row['image_id'],source_parent_id=row['source_parent_id'],original_valid_parent_id=row['original_valid_parent_id'],
        mutation_family_id=row['mutation_family_id'],source_lineage=row['source_lineage'],split=row['dataset_split'],
        image_file_path=row['image_file_path'],image_sha256=row['image_sha256'],validity=row['validity_status'],
        invalidity_category=row['invalidity_category'],paired_mutation_category=paired_category,canonical_rule_id=row['canonical_rule_id'],
        ground_truth_verification_evidence=row['verification_evidence'],mutation_parameters=row.get('mutation_parameters',{}),
        augmentation_recipe=row['augmentation_recipe'],human_review_status=row.get('human_review_status','NOT_REVIEWED'),
        readability=row.get('readability','UNDECLARED'),source_annotation_path='annotations/'+row['image_id']+'.json',
        expected_outcome=row['expected_system_outcome'],actual_outcome=decision['outcome'],raw_OCR_transcription=original,
        interpreted_transcription=interpreted,valid_reference_transcription=truth,raw_errors=page_errors(truth,original),
        interpreted_errors=page_errors(truth,interpreted),raw_decision=result['raw_f1_decision'],decision=decision,
        triggered_reason_codes=decision['reason_codes'],original_reason_codes=result['original_reason_codes'],
        original_evidence=result['original_evidence'],interpretations=result['interpretations'],corrections=corrections,
        correction_assessments=assessments,interpreted_syntax=validate_tiers(interpreted),
        syntax_failure_resolved=result['syntax_failure_resolved'],interpretation_uncertain=result['uncertain'],
        segmentation_result={k:obs.get(k) for k in ('preprocessing_status','segmentation_accepted','segmentation_reason','segmentation_rows','diagnostics','ocr_status')},
        preprocessing_failure=preprocessing_failure,segmentation_failure=segmentation_failure,ocr_failure=bool(ocr_failure),pipeline_exception=exception,
        confidence=obs.get('confidence',{}),matched_valid_control_id=row['matched_valid_control_id'],
        variant='illuminated' if row['augmentation_recipe'] is not None else 'clean',
        length_group='short_le4' if max_length<=4 else 'medium_5_to64' if max_length<65 else 'long_ge65',
        row_group='single_row' if len(row['raw_display'])==1 else 'multi_row')


def evaluate_records(raw_items,dataset,candidate):
    categories={pid:next(r['invalidity_category'] for r in group if r['validity_status']=='INVALID') for pid,group in dataset.groups.items()}
    return [evaluate_case(item,dataset.by_id[item['identity']['image_id']],candidate,categories[item['identity']['source_parent_id']]) for item in raw_items]


def load_candidate(directory,dataset):
    directory=Path(directory);done=load_json(directory/'completion.json')
    if not done.get('complete') or digest(directory/'results.json')!=done['results_sha256']:raise ValueError('Candidate output incomplete or changed')
    result=load_json(directory/'results.json');rows=result['rows']
    if done['candidate']!=result['candidate'] or result['candidate'] not in ('R0','R1'):raise ValueError('Candidate marker differs')
    if result['lineage']['experiment_lock_sha256']!=digest(ROOT/'configs/experiment_lock.json') or result['lineage']['checkpoint_sha256']!=CHECKPOINT_SHA:
        raise ValueError('Candidate frozen code/checkpoint lineage differs')
    if (result['lineage']['dataset_manifest_sha256']!=dataset.manifest_sha256 or result['lineage']['selection_sha256']!=dataset.selection_sha256
            or len(rows)!=len(dataset.rows) or done['records']!=len(rows)):raise ValueError('Candidate development coverage/lineage differs')
    if {r['image_id'] for r in rows}!=set(dataset.by_id):raise ValueError('Candidate record set is not exactly development')
    for r in rows:
        source=dataset.by_id[r['image_id']]
        if (r['split']!=SPLIT or r['source_parent_id']!=source['source_parent_id'] or r['image_file_path']!=source['image_file_path']
                or r['image_sha256']!=source['image_sha256']):raise ValueError('Candidate identity/path/parent changed')
        for output_key,source_key in [('validity','validity_status'),('invalidity_category','invalidity_category'),('expected_outcome','expected_system_outcome'),
                                      ('canonical_rule_id','canonical_rule_id'),('matched_valid_control_id','matched_valid_control_id')]:
            if r[output_key]!=source[source_key]:raise ValueError('Candidate evaluation-label drift')
        if r['valid_reference_transcription']!=expected_valid_rows(source):raise ValueError('Candidate legitimate reference differs')
        if source['validity_status']=='INVALID' and (r['raw_errors'] is not None or r['interpreted_errors'] is not None):raise ValueError('Invalid-image CER is forbidden')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path);p.add_argument('--split',choices=(SPLIT,),default=SPLIT)
    p.add_argument('--raw-dir',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();verify_integrity();dataset=DevelopmentDataset(a.dataset,a.split)
    config=load_json(a.config);candidate=config.get('candidate')
    if candidate not in ('R0','R1') or config.get('tier_aware')!=(candidate=='R1'):raise ValueError('Invalid R0/R1 configuration')
    expected=ROOT/('configs/r0_baseline.yaml' if candidate=='R0' else 'configs/r1_tier_aware.yaml')
    if digest(a.config)!=digest(expected):raise ValueError('Configuration is not frozen')
    raw,lineage=load_raw_run(a.raw_dir,dataset)
    if lineage['plan']['experiment_lock_sha256']!=digest(ROOT/'configs/experiment_lock.json'):raise ValueError('Raw run was made with different evaluation/inference code')
    rows=evaluate_records(raw,dataset,candidate);summary,parents,groups=grouped(rows)
    out=new_output(a.output_dir)
    provenance=dict(dataset_manifest_sha256=dataset.manifest_sha256,selection_sha256=dataset.selection_sha256,
        raw_observations_sha256=lineage['completion']['raw_sha256'],raw_plan_sha256=lineage['completion']['plan_sha256'],
        experiment_lock_sha256=digest(ROOT/'configs/experiment_lock.json'),checkpoint_sha256=lineage['plan']['checkpoint_sha256'])
    write_json(out/'results.json',dict(candidate=candidate,config=config,lineage=provenance,rows=rows,
        scope='development only; synthetic canonical labels, no human expert review; no independent test claim'))
    write_json(out/'summary.json',summary);write_json(out/'by_parent.json',parents);write_json(out/'by_group.json',groups)
    write_csv(out/'per_image.csv',rows)
    write_json(out/'completion.json',dict(complete=True,candidate=candidate,records=len(rows),results_sha256=digest(out/'results.json')))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
