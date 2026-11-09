"""Separate explicitly authorized loader. Original DevelopmentDataset untouched."""
from collections import Counter,defaultdict
from pathlib import Path
from release_io import RELEASE,load,sha,canonical

SPLITS=('validation_calibration','protected_test')
EXPECTED={'development_design':(2800,700),'validation_calibration':(600,150),'protected_test':(600,150)}


def audit_ownership(rows,parents,expected=EXPECTED):
    ids=set();owners={k:{} for k in ('source_parent_id','image_sha256')};groups=defaultdict(list);by_parent={}
    for p in parents:
        if p['source_parent_id'] in by_parent:raise ValueError('Duplicate parent metadata')
        by_parent[p['source_parent_id']]=p
    semantic={}
    for r in rows:
        split=r['dataset_split'];id=r['image_id'];parent=r['source_parent_id']
        if split not in expected or id in ids:raise ValueError('Unknown split or duplicate sample ID')
        ids.add(id)
        for key in owners:
            value=r[key]
            if value in owners[key] and owners[key][value]!=split:raise ValueError('Cross-split '+key)
            owners[key][value]=split
        if parent not in by_parent or by_parent[parent]['dataset_split']!=split:raise ValueError('Parent metadata/ownership mismatch')
        key=by_parent[parent]['semantic_key']
        if key in semantic and semantic[key]!=split:raise ValueError('Semantic source parent crosses split')
        semantic[key]=split;groups[split].append(r)
    summary={}
    for split,(count,nparents) in expected.items():
        selected=groups[split];ps={r['source_parent_id'] for r in selected}
        if len(selected)!=count or len(ps)!=nparents:raise ValueError('Incomplete split: '+split)
        summary[split]=dict(images=len(selected),parents=len(ps),validity=dict(Counter(r['validity_status'] for r in selected)))
    return summary


def select_records(rows,split):
    if split not in SPLITS:raise ValueError('Explicit authorized held-out split required')
    return sorted([r for r in rows if r['dataset_split']==split],key=lambda r:r['image_id'])


def safe_image_path(directory,row,split):
    if split not in SPLITS or row['dataset_split']!=split:raise ValueError('Image outside selected held-out split')
    rel=Path(row['image_file_path']);root=Path(directory).resolve()
    if rel.is_absolute() or '..' in rel.parts:raise ValueError('Unsafe dataset path')
    path=(root/rel).resolve()
    if not path.is_relative_to(root):raise ValueError('Image escapes frozen dataset')
    return path


def identity(row):return {k:row[k] for k in ('image_id','image_sha256','source_parent_id','original_valid_parent_id','mutation_family_id','dataset_split','source_lineage')}


def rendering_metadata(row):
    # Exact original helper inputs, stripped BEFORE entering frozen runtime.
    layout=row['layout'];recipe=row['augmentation_recipe']
    return dict(layout={k:layout.get(k) for k in ('font_sha256','font_size_px','convention')},augmentation_recipe=None if recipe is None else {k:recipe.get(k) for k in ('padding','operations','output_format')})


class HeldoutDataset:
    def __init__(self,manifest,split):
        if split not in SPLITS:raise ValueError('Only explicit validation_calibration/protected_test permitted')
        d=manifest['dataset'];self.directory=Path(d['original_directory']).resolve();self.split=split
        # Read only metadata here: no image bytes or annotation files.
        for name,h in d['original_metadata_sha256'].items():
            if sha(self.directory/name)!=h:raise ValueError('Actual dataset metadata drift: '+name)
        rows=load(RELEASE/d['manifest_snapshot']);parents=load(RELEASE/d['parents_snapshot'])
        self.audit=audit_ownership(rows,parents);self.rows=select_records(rows,split);self.by_id={r['image_id']:r for r in self.rows}
        self.groups=defaultdict(list)
        seal=load(RELEASE/d['protected_seal_snapshot'])
        if not seal.get('frozen'):raise ValueError('Protected dataset is not sealed')
        for r in self.rows:
            id=r['image_id']
            if any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_' for c in id):raise ValueError('Unsafe image ID')
            if r['verification_status']!='AUTO_VERIFIED' or r.get('ambiguity_or_exclusion_reason'):raise ValueError('Unverified/excluded sample')
            if r['validity_status'] not in ('VALID','INVALID'):raise ValueError('Unknown offline label')
            if split=='protected_test' and seal['files'].get(r['image_file_path'])!=r['image_sha256']:raise ValueError('Protected seal/manifest image metadata mismatch')
            if not safe_image_path(self.directory,r,split).is_file():raise FileNotFoundError(id)
            self.groups[r['source_parent_id']].append(r)
        for group in self.groups.values():
            if Counter((r['validity_status'],r['augmentation_recipe'] is not None) for r in group)!=Counter({('VALID',False):1,('VALID',True):1,('INVALID',False):1,('INVALID',True):1}):raise ValueError('Incomplete parent quartet')
        self.manifest_sha256=d['manifest_sha256'];self.selection_sha256=canonical([identity(r) for r in self.rows])

    def image_bytes(self,id):
        r=self.by_id[id];raw=safe_image_path(self.directory,r,self.split).read_bytes()
        import hashlib
        if hashlib.sha256(raw).hexdigest()!=r['image_sha256']:raise ValueError('Image SHA mismatch: '+id)
        return raw
