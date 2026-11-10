"""Strict manifest-only selection. Development commands never access other images."""
from collections import Counter,defaultdict
from integration import *

SPLIT='development_design'
CATEGORIES={'wrong_tier':'CANON_SYMBOL_DIRECTION','invalid_count_token':'CANON_COUNT_ENCODING','missing_count':'CANON_REQUIRED_SYMBOL_COUNT'}


def select_records(rows,split=SPLIT):
    if split!=SPLIT:raise ValueError('Only development_design is permitted; validation and protected_test are blocked')
    ids=set();ownership={};hash_splits={}
    for r in rows:
        if not isinstance(r,dict) or {'image_id','source_parent_id','dataset_split','image_sha256'}-set(r):raise ValueError('Manifest split schema missing')
        id=r['image_id'];parent=r['source_parent_id'];part=r['dataset_split']
        if id in ids:raise ValueError('Duplicate manifest image ID')
        ids.add(id)
        if part not in (SPLIT,'validation_calibration','protected_test'):raise ValueError('Unknown split identifier')
        if parent in ownership and ownership[parent]!=part:raise ValueError('Parent crosses split')
        ownership[parent]=part
        h=r['image_sha256']
        if h in hash_splits and hash_splits[h]!=part:raise ValueError('Byte-identical images cross split')
        hash_splits[h]=part
    # No images, annotation files or source structures from excluded splits
    # are touched here. Only ownership/hash/ID metadata is examined globally.
    return sorted([r for r in rows if r['dataset_split']==SPLIT],key=lambda r:r['image_id'])


def safe_image_path(dataset,row):
    if row['dataset_split']!=SPLIT:raise ValueError('Image access outside development is forbidden')
    relative=Path(row['image_file_path'])
    if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe image path')
    path=(Path(dataset)/relative).resolve()
    if not path.is_relative_to(Path(dataset).resolve()):raise ValueError('Image path escapes declared dataset')
    return path


def image_bytes(dataset,row):
    path=safe_image_path(dataset,row)
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=row['image_sha256']:raise ValueError('Image hash mismatch: '+row['image_id'])
    return raw


def expected_valid_rows(row):
    if row['validity_status']!='VALID':return None
    return [{('count' if role=='count_line' else role):text.lstrip(' ') for role,text in r.items() if text.strip()}
            for r in row['raw_display']]


def validate_selected(rows,parents,history_keys,reserved_keys):
    by={r['image_id']:r for r in rows};groups=defaultdict(list)
    required={'image_id','image_file_path','image_sha256','source_parent_id','original_valid_parent_id','mutation_family_id',
        'validity_status','expected_system_outcome','invalidity_category','canonical_rule_id','verification_status',
        'verification_evidence','original_structured_notation','raw_display','layout','original_image_hash',
        'matched_valid_control_id','augmentation_recipe','dataset_split','source_lineage','ctc_transcription','semantic_validity'}
    seen_hashes=set()
    for r in rows:
        if required-set(r):raise ValueError('Incompatible dataset-v2 schema')
        id=r['image_id'];pid=r['source_parent_id'];status=r['validity_status']
        if not id or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_' for c in id):raise ValueError('Unsafe record identifier')
        if r['dataset_split']!=SPLIT or pid not in parents:raise ValueError('Selected parent missing or forbidden')
        if status not in ('VALID','INVALID') or r['semantic_validity']!=status:raise ValueError('Unsupported or conflicting validity label')
        if r['verification_status']!='AUTO_VERIFIED' or r.get('ambiguity_or_exclusion_reason'):
            raise ValueError('Excluded, quarantined or unverified example')
        if r['verification_evidence'].get('classification')!=status or r['verification_evidence'].get('status')!='AUTO_VERIFIED':raise ValueError('Structural evidence and label disagree')
        if r['expected_system_outcome']!=('ACCEPT' if status=='VALID' else 'REJECT_INVALID'):raise ValueError('Expected outcome and validity disagree')
        if status=='INVALID':
            if r['invalidity_category'] not in CATEGORIES or r['canonical_rule_id']!=CATEGORIES[r['invalidity_category']]:raise ValueError('Unsupported mutation category/rule')
            if r['ctc_transcription'] is not None:raise ValueError('Invalid image has fabricated CTC target')
        elif r['invalidity_category']!='none' or r['ctc_transcription']!=rules.canonical_text(r['original_structured_notation']):
            raise ValueError('Valid control/reference inconsistency')
        if status=='VALID' and r['raw_display']!=[dict(zip(('upper','count_line','lower'),rules.canonical_row(x['runs']))) for x in r['original_structured_notation']['logical_rows']]:
            raise ValueError('Valid raw notation differs from canonical reference')
        if r['mutation_family_id']!=pid or parents[pid]['dataset_split']!=SPLIT:raise ValueError('Lineage ownership mismatch')
        key=rules.semantic_key(r['original_structured_notation'])
        if parents[pid]['semantic_key']!=key or pid!='v2p_'+key[:24]:raise ValueError('Canonical source parent mismatch')
        if key in history_keys or key in reserved_keys:raise ValueError('Historical/preflight source overlap')
        lineage=r['source_lineage']
        if (lineage.get('historical_parent_reused') is not False or lineage.get('exact_semantic_overlap_found') is not False
                or lineage.get('historical_overlap_coverage')!='full' or lineage.get('known_historical_semantic_novelty_verified') is not True):
            raise ValueError('Historical source novelty not established')
        control=by.get(r['matched_valid_control_id'])
        if (not control or control['validity_status']!='VALID' or control['augmentation_recipe'] is not None
                or control['source_parent_id']!=pid or control['dataset_split']!=SPLIT):raise ValueError('Matched valid control missing/wrong lineage')
        if r['original_image_hash']!=control['image_sha256']:raise ValueError('Original valid image hash mismatch')
        if r['image_sha256'] in seen_hashes:raise ValueError('Duplicate development image bytes')
        seen_hashes.add(r['image_sha256']);groups[pid].append(r)
    for pid,g in groups.items():
        if len(g)!=4 or Counter((r['validity_status'],r['augmentation_recipe'] is not None) for r in g)!=Counter({('VALID',False):1,('VALID',True):1,('INVALID',False):1,('INVALID',True):1}):
            raise ValueError('Expected one valid/invalid clean+illumination quartet per parent')
        if len({r['invalidity_category'] for r in g if r['validity_status']=='INVALID'})!=1:raise ValueError('Mutation family category mismatch')
    return groups


class DevelopmentDataset:
    def __init__(self,directory=None,split=SPLIT,verify_files=False,*,spec=None):
        if split!=SPLIT:raise ValueError('Only development_design is allowed')
        # `spec` is dependency injection for tiny unit fixtures, never a CLI bypass.
        spec=spec or load_json(ROOT/'configs/dataset.json')
        self.directory=Path(directory or spec['dataset_dir']).resolve()
        if self.directory!=Path(spec['dataset_dir']).resolve():raise ValueError('Dataset differs from explicitly frozen full dataset')
        if any('quarantined' in p.name.lower() or (p/'QUARANTINED.json').exists() for p in (self.directory,*self.directory.parents)):
            raise ValueError('Quarantined dataset is forbidden')
        complete=load_json(self.directory/'completion.json')
        if not complete.get('complete') or complete.get('mode')!='full' or complete.get('history_coverage')!='full':raise ValueError('Dataset is not a completed full v2 run')
        for name,h in spec['metadata_hashes'].items():
            if digest(self.directory/name)!=h:raise ValueError('Frozen dataset metadata drift: '+name)
        if digest(self.directory/'manifest.json')!=complete['manifest_sha256']:raise ValueError('Dataset manifest/completion mismatch')
        history_path=Path(spec['history_index'])
        if digest(history_path)!=complete['history_index_sha256']:raise ValueError('History index lineage mismatch')
        history=load_json(history_path)
        if not history['full_historical_coverage'] or history['errors']:raise ValueError('Historical coverage incomplete')
        rows=load_json(self.directory/'manifest.json')
        if len(rows)!=spec['total_records']:raise ValueError('Full dataset record count differs from frozen metadata')
        self.rows=select_records(rows,split)
        if len(self.rows)!=spec['development_records']:raise ValueError('Development coverage incomplete')
        parent_meta=load_json(self.directory/'parents.json')
        parents={p['source_parent_id']:p for p in parent_meta if p['dataset_split']==SPLIT}
        if len(parents)!=spec['development_parents']:raise ValueError('Development source-parent count mismatch')
        reserved=set()
        for path in spec.get('reserved_parent_indexes',[]):reserved.update(p['semantic_key'] for p in load_json(path))
        self.groups=validate_selected(self.rows,parents,set(history['semantic_keys']),reserved)
        self.by_id={r['image_id']:r for r in self.rows}
        self.manifest_sha256=complete['manifest_sha256']
        self.selection_sha256=content_hash([self.identity(r) for r in self.rows])
        for r in self.rows:
            path=safe_image_path(self.directory,r)
            if not path.is_file():raise FileNotFoundError(path)
            if verify_files:
                image_bytes(self.directory,r)
                if load_json(self.directory/'annotations'/(r['image_id']+'.json'))!=r:raise ValueError('Development annotation/manifest mismatch')

    @staticmethod
    def identity(row):
        return {k:row[k] for k in ('image_id','image_sha256','source_parent_id','original_valid_parent_id','mutation_family_id','dataset_split','source_lineage')}
