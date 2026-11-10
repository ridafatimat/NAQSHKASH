"""Generate only into a NEW v2 run. Preflight is bounded; full mode is manual."""
import argparse
import copy
import random
from collections import Counter
from common_v2 import *
from render_invalid import layout_for, render_display, pixel_difference
from mutations import mutate
from independent_verify import verify_image, display_agrees, verify_matched_negative, visual_aliases
from create_splits import assign_splits, verify_splits
from scan_variants import make_variant


def make_parent(namespace,index,seed):
    value=int(hash_object([namespace,seed,index])[:16],16)
    rng=random.Random(value)
    upper=list(rules.EXCLUSIVE_UP_SYMBOLS);lower=list(rules.EXCLUSIVE_DOWN_SYMBOLS)
    if index%3==0:
        # Wrong-tier parents deliberately use only visibly distinctive shapes;
        # counts/other categories still cover the full canonical symbol pool.
        upper=[c for c in upper if c not in visual_aliases()]
        lower=[c for c in lower if c not in visual_aliases()]
    symbols=sorted(rng.sample(upper,4)+rng.sample(lower,4))
    if index%4==0:symbols=sorted(set(upper)|set(lower))
    row_count=1+(index//3)%3
    width=[47,63,96,128,189,259,320,400][index%8]
    kwargs={}
    if index%7==0:
        width=128;patterns=[]
        for _ in range(row_count):
            pattern=[];previous=None
            for n in [11,19]+[1]*(width-30):
                symbol=rng.choice([s for s in symbols if s!=previous])
                pattern.append((symbol,n));previous=symbol
            patterns.append(pattern)
        kwargs=dict(run_pattern='controlled_rows',controlled_row_patterns_by_row=patterns)
    sample=generate_talim(f'{namespace}_{index:06d}',row_count,width,symbols,seed,value,
        max_run_length=259,required_counts=() if kwargs else (11,19),**kwargs).to_metadata()
    if check_sample(sample):raise ValueError('Canonical source generator returned invalid data')
    return sample,value


def source_plan(config,mode,count,history):
    namespace=config['preflight_namespace'] if mode=='preflight' else config['full_namespace']
    keys=set(history['semantic_keys'])
    reserved=set()
    preflight=ROOT/'dataset/preflight/parents.json'
    if mode=='full' and preflight.is_file():reserved={p['semantic_key'] for p in read_json(preflight)}
    if mode=='full':
        for path in (ROOT/'dataset').glob('quarantined_preflight_*/parents.json'):
            reserved.update(p['semantic_key'] for p in read_json(path))
    plans=[];seen=set();excluded=[]
    for i in range(count):
        parent,parent_seed=make_parent(namespace,i,config['seed'])
        key=rules.semantic_key(parent)
        if key in keys or key in reserved or key in seen:
            excluded.append(dict(index=i,status='EXCLUDED',reason='historical/preflight/within-run semantic duplicate',semantic_key=key))
            continue  # do NOT backfill quota with weak/ambiguous examples
        seen.add(key)
        plans.append(dict(source_parent_id='v2p_'+key[:24],semantic_key=key,seed=parent_seed,
            original_valid_parent_id=parent['sample_id'],original=parent,index=i,
            category=config['categories'][i%len(config['categories'])],
            source_lineage=dict(generator='existing canonical generate_talim',namespace=namespace,seed=parent_seed,
                historical_parent_reused=False,exact_semantic_overlap_found=False,
                historical_overlap_coverage=history['mode'],independent_unseen_claim=False,
                known_historical_semantic_novelty_verified=mode=='full' and history['full_historical_coverage'])))
    return plans,excluded


def generate(config,mode,output,history_path):
    if config['schema_version']!=1 or config['categories']!=list(RULES) or config['variants_per_clean_image']!=1:
        raise ValueError('Unsupported frozen generation design')
    verify_design();verify_historical()
    history=read_json(history_path)
    if history.get('errors'):raise ValueError('History index has unresolved source errors')
    if mode=='full':
        if not history.get('full_historical_coverage'):raise ValueError('Full generation requires complete known historical semantic coverage')
        for value in history['archives'].values():
            if digest(value['path'])!=value['sha256']:raise ValueError('Historical archive changed since indexing')
    count=config['preflight_per_category']*len(config['categories']) if mode=='preflight' else config['full_parents']
    if mode=='preflight' and count>30:raise ValueError('Preflight safety cap: at most30 source parents')
    plans,excluded=source_plan(config,mode,count,history)
    assignments=assign_splits([p['source_parent_id'] for p in plans],config['seed'],mode=='preflight')
    out=new_output(output)
    for directory in ('valid_controls','invalid_talim','scanned_variants','annotations','parents','reports'):
        (out/directory).mkdir()
    rows=[];image_hashes={};parents=[]
    for plan in plans:
        sample=plan['original'];pid=plan['source_parent_id'];cat=plan['category'];split=assignments[pid]
        layout=layout_for(sample)
        valid_id=pid+'_valid';invalid_id=pid+'_'+cat
        valid_path=out/'valid_controls'/(valid_id+'.png')
        render_talim_text(rules.canonical_text(sample),valid_path,FONT)
        valid=Image.open(valid_path);valid.load()
        if valid.tobytes()!=render_display(displayed_rows(sample),layout).tobytes():
            raise ValueError('FUNDAMENTAL: isolated rendering does not reproduce canonical valid control')
        valid_proof=verify_image(valid,layout)
        if valid_proof['classification']!='VALID':raise ValueError('FUNDAMENTAL: canonical positive failed independent raster parse')
        modified,parameters=mutate(sample,cat,layout,plan['index'])
        invalid=render_display(modified,layout)
        diff=pixel_difference(valid,invalid)
        proof=verify_matched_negative(valid,invalid,layout,modified,RULES[cat])
        if (not diff['changed_pixels'] or proof['status']!='AUTO_VERIFIED' or proof['classification']!='INVALID' or RULES[cat] not in proof['reason_codes']
                or not display_agrees(invalid,layout,modified)):
            write_json(out/'reports/fundamental_failure.json',dict(parent=pid,category=cat,proof=proof,difference=diff,status='EXCLUDED'))
            raise ValueError('FUNDAMENTAL: invalidity or visible rendering verification failed; stop, do not fill quota')
        invalid_path=out/'invalid_talim'/(invalid_id+'.png');invalid.save(invalid_path)
        parent_file=out/'parents'/(pid+'.json');write_json(parent_file,dict(plan,dataset_split=split,layout=layout))
        parents.append(dict(source_parent_id=pid,semantic_key=plan['semantic_key'],dataset_split=split,
                            original_valid_parent_id=sample['sample_id'],source_lineage=plan['source_lineage']))
        for status,id,path,display,verification in [('VALID',valid_id,valid_path,displayed_rows(sample),valid_proof),
                                                   ('INVALID',invalid_id,invalid_path,modified,proof)]:
            row=dict(image_id=id,image_file_path=str(path.relative_to(out)),image_sha256=digest(path),source_parent_id=pid,
                original_valid_parent_id=sample['sample_id'],mutation_family_id=pid,validity_status=status,
                expected_system_outcome='ACCEPT' if status=='VALID' else 'REJECT_INVALID',
                invalidity_category='none' if status=='VALID' else cat,canonical_rule_id='CANONICAL_VALID' if status=='VALID' else RULES[cat],
                mutation_parameters={} if status=='VALID' else parameters,verification_status=verification['status'],verification_evidence=verification,
                original_structured_notation=sample,
                mutated_structured_notation=dict(format='raw_three_tier_display',rows=display,no_numeric_ctc_target=status=='INVALID'),
                raw_display=display,layout=layout,original_image_hash=digest(valid_path),pixel_difference_evidence=diff if status=='INVALID' else {},
                matched_valid_control_id=valid_id,augmentation_recipe=None,dataset_split=split,source_lineage=plan['source_lineage'],
                human_review_status='NOT_REVIEWED',ambiguity_or_exclusion_reason='',ctc_transcription=rules.canonical_text(sample) if status=='VALID' else None,
                semantic_validity=status,readability='READABLE_SYNTHETIC_CANONICAL_FONT')
            rows.append(row)
            write_json(out/'annotations'/(id+'.json'),row)
            raw,scan,recipe=make_variant(valid if status=='VALID' else invalid,plan['seed'])
            scan_proof=verify_image(scan,layout,recipe)
            same=scan_proof['classification']==status and display_agrees(scan,layout,display,recipe)
            scan_id=id+'_illumination';scan_path=out/'scanned_variants'/(scan_id+'.png');scan_path.write_bytes(raw)
            scanrow=copy.deepcopy(row)
            scanrow.update(image_id=scan_id,image_file_path=str(scan_path.relative_to(out)),image_sha256=digest(scan_path),
                augmentation_recipe=recipe,verification_status=scan_proof['status'] if same else 'NEEDS_DOMAIN_REVIEW',
                verification_evidence=scan_proof,validity_status=status if same else 'AMBIGUOUS',
                semantic_validity=status if same else 'UNCONFIRMED',expected_system_outcome=row['expected_system_outcome'] if same else 'REVIEW_UNCERTAIN',
                ambiguity_or_exclusion_reason='' if same else 'augmentation did not preserve independently verified image evidence',
                ctc_transcription=row['ctc_transcription'] if same else None)
            rows.append(scanrow);write_json(out/'annotations'/(scan_id+'.json'),scanrow)
        print('Verified source family',len(parents),'/',len(plans),cat,flush=True)
    # Fail closed on ANY pixel duplicate, not merely cross-split duplicates.
    duplicates=[]
    for row in rows:
        h=row['image_sha256']
        if h in image_hashes:duplicates.append([image_hashes[h],row['image_id']])
        image_hashes[h]=row['image_id']
    if duplicates:
        write_json(out/'reports/duplicate_failure.json',duplicates)
        raise ValueError('Duplicate rasters detected; run remains incomplete and unsealed')
    split_report=verify_splits(rows)
    write_json(out/'parents.json',parents)
    write_json(out/'manifest.json',rows);write_csv(out/'manifest.csv',rows)
    write_json(out/'reports/excluded_sources.json',excluded)
    counts=dict(mode=mode,source_parents=len(parents),images=len(rows),
        clean_invalid=sum(r['validity_status']=='INVALID' and r['augmentation_recipe'] is None for r in rows),
        clean_valid_controls=sum(r['validity_status']=='VALID' and r['augmentation_recipe'] is None for r in rows),
        scanned_style_images=sum(r['augmentation_recipe'] is not None for r in rows),
        independently_verified_invalid_images=sum(r['validity_status']=='INVALID' and r['verification_status']=='AUTO_VERIFIED' for r in rows),
        domain_review_images=sum(r['verification_status']=='NEEDS_DOMAIN_REVIEW' for r in rows),
        validity_counts=dict(Counter(r['validity_status'] for r in rows)),
        clean_invalid_by_category=dict(Counter(r['invalidity_category'] for r in rows if r['validity_status']=='INVALID' and r['augmentation_recipe'] is None)),
        split_source_parents=dict(Counter(p['dataset_split'] for p in parents)),human_reviewed_images=0,
        excluded_parent_candidates=len(excluded),duplicate_images=duplicates,**{k:v for k,v in split_report.items() if k!='source_parents'})
    write_json(out/'reports/generation_summary.json',counts)
    from validate_dataset import validate
    validation=validate(out,construction=True)
    write_json(out/'reports/construction_validation.json',validation)
    from gallery import create_gallery
    create_gallery(out,out/'gallery.html')
    if mode=='full':
        protected={str((out/r['image_file_path']).relative_to(out)):r['image_sha256'] for r in rows if r['dataset_split']=='protected_test'}
        protected.update({str((out/'annotations'/(r['image_id']+'.json')).relative_to(out)):digest(out/'annotations'/(r['image_id']+'.json')) for r in rows if r['dataset_split']=='protected_test'})
        protected.update({str((out/'parents'/(p['source_parent_id']+'.json')).relative_to(out)):digest(out/'parents'/(p['source_parent_id']+'.json')) for p in parents if p['dataset_split']=='protected_test'})
        write_json(out/'protected_test_seal.json',dict(frozen=True,files=protected,
            policy='Construction verification completed before sealing. Later validation is hash-only for protected examples; no test gallery or semantic debugging.'))
    verify_historical();verify_design()
    write_json(out/'completion.json',dict(complete=True,mode=mode,manifest_sha256=digest(out/'manifest.json'),
        history_index_sha256=digest(history_path),history_coverage=history['mode'],
        design_lock_sha256=digest(ROOT/'configs/design_lock.json'),counts=counts,
        independent_source_claim_scope='new seeded canonical parents; exact semantic exclusion within indexed archives, not independent real-world evidence'))
    return counts


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/generation_v2.yaml')
    p.add_argument('--mode',choices=('preflight','full'),required=True)
    p.add_argument('--history-index',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args()
    if digest(a.config)!=digest(ROOT/'configs/generation_v2.yaml'):raise ValueError('Use the frozen generation configuration')
    print(json.dumps(generate(read_json(a.config),a.mode,a.output_dir,a.history_index),indent=2))


if __name__=='__main__':main()
