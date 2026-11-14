"""Dataset-v2 QA; sealed protected examples receive HASH checks only."""
import argparse
from common_v2 import *
from independent_verify import verify_image,display_agrees
from render_invalid import pixel_difference
from create_splits import verify_splits


def validate(dataset,construction=False):
    dataset=Path(dataset)
    if (dataset/'QUARANTINED.json').exists():raise ValueError('Quarantined provisional dataset; not eligible for training/evaluation')
    rows=read_json(dataset/'manifest.json')
    if not construction:
        complete=read_json(dataset/'completion.json')
        if not complete['complete'] or digest(dataset/'manifest.json')!=complete['manifest_sha256']:raise ValueError('Dataset incomplete or manifest drift')
    if len({r['image_id'] for r in rows})!=len(rows):raise ValueError('Duplicate IDs')
    if len({r['image_sha256'] for r in rows})!=len(rows):raise ValueError('Duplicate image bytes')
    parent_records={p['source_parent_id']:p for p in read_json(dataset/'parents.json')}
    parent_ids=set(parent_records)
    controls={r['image_id']:r for r in rows if r['augmentation_recipe'] is None and r['validity_status']=='VALID'}
    seal=read_json(dataset/'protected_test_seal.json') if (dataset/'protected_test_seal.json').exists() else None
    if construction and seal:raise ValueError('Protected split already sealed: semantic reconstruction forbidden')
    for name,h in (seal['files'].items() if seal else []):
        if digest(dataset/name)!=h:raise ValueError('Protected test artifact drift')
    checked=protected=0
    for row in rows:
        path=(dataset/row['image_file_path']).resolve()
        if not path.is_relative_to(dataset.resolve()):raise ValueError('Unsafe image path')
        if digest(path)!=row['image_sha256']:raise ValueError('Image hash mismatch')
        if row['source_parent_id'] not in parent_ids or row['mutation_family_id']!=row['source_parent_id']:raise ValueError('Lineage mismatch')
        if row['dataset_split']=='protected_test' and seal:
            protected+=1;continue
        if read_json(dataset/'annotations'/(row['image_id']+'.json'))!=row:raise ValueError('Image annotation and manifest disagree')
        if check_sample(row['original_structured_notation']):raise ValueError('Invalid original valid parent')
        key=rules.semantic_key(row['original_structured_notation'])
        if key!=parent_records[row['source_parent_id']]['semantic_key'] or row['source_parent_id']!='v2p_'+key[:24]:raise ValueError('Canonical semantic parent ownership mismatch')
        source=read_json(dataset/'parents'/(row['source_parent_id']+'.json'))
        if source['original']!=row['original_structured_notation']:raise ValueError('Original parent source/annotation mismatch')
        if row['validity_status']=='INVALID' and row['ctc_transcription'] is not None:raise ValueError('Invalid image has fake CTC target')
        control=controls[row['matched_valid_control_id']]
        if control['source_parent_id']!=row['source_parent_id'] or control['dataset_split']!=row['dataset_split']:raise ValueError('Matched control lineage/split mismatch')
        with Image.open(path) as im:
            im.load();proof=verify_image(im,row['layout'],row['augmentation_recipe'])
            if row['verification_status']=='AUTO_VERIFIED':
                if proof['classification']!=row['validity_status']:raise ValueError('Independent validity disagreement')
                if not display_agrees(im,row['layout'],row['raw_display'],row['augmentation_recipe']):raise ValueError('Declared glyphs do not match raster')
            elif row['validity_status']=='INVALID':raise ValueError('Unproven sample entered invalid-image collection')
            if row['validity_status']=='INVALID' and row['augmentation_recipe'] is None:
                with Image.open(dataset/control['image_file_path']) as valid:
                    diff=pixel_difference(valid,im)
                if not diff['changed_pixels']:raise ValueError('Annotation-only negative forbidden')
                if diff!=row['pixel_difference_evidence']:raise ValueError('Pixel-difference evidence mismatch')
                if row['canonical_rule_id'] not in proof['reason_codes']:raise ValueError('Intended verified rule not found independently')
        checked+=1
    report=verify_splits(rows)
    return dict(images=len(rows),semantic_images_checked=checked,protected_hash_only_images=protected,
                annotations_consistent=True,duplicates=0,**report)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--report-dir',type=Path)
    a=p.parse_args();verify_historical();verify_design();result=validate(a.dataset)
    if a.report_dir:
        out=new_output(a.report_dir);write_json(out/'validation.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
