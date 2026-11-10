"""Deterministic source-family ownership assigned BEFORE mutation/augmentation."""
import hashlib


def assign_splits(parent_ids,seed,preflight=False):
    if len(parent_ids)!=len(set(parent_ids)):raise ValueError('Duplicate source-family IDs')
    if preflight:return {p:'development_design' for p in parent_ids}
    ranked=sorted(parent_ids,key=lambda p:hashlib.sha256(f'v2split:{seed}:{p}'.encode()).hexdigest())
    n=len(ranked);nt=round(n*.15);nv=round(n*.15)
    return {p:('protected_test' if i<nt else 'validation_calibration' if i<nt+nv else 'development_design') for i,p in enumerate(ranked)}


def verify_splits(rows):
    parents={};families={};hashes={}
    for r in rows:
        for lookup,key in ((parents,r['source_parent_id']),(families,r['mutation_family_id']),(hashes,r['image_sha256'])):
            if key in lookup and lookup[key]!=r['dataset_split']:raise ValueError('Cross-split lineage/hash leakage: '+key)
            lookup[key]=r['dataset_split']
    return dict(source_parents=len(parents),parent_disjoint=True,mutation_family_disjoint=True,byte_hash_disjoint=True)
