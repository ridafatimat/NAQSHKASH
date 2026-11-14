"""Resumable per-record immutable journal; one sealed JSONL feeds both candidates."""
import contextlib
import tempfile
from integration import *


@contextlib.contextmanager
def exclusive_run_lock(path):
    stream=Path(path).open('a+b')
    stream.seek(0,2)
    if not stream.tell():stream.write(b'0');stream.flush()
    stream.seek(0)
    try:
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except OSError:
        stream.close();raise RuntimeError('This inference run is already active')
    try:yield
    finally:
        stream.seek(0)
        if os.name=='nt':msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
        else:fcntl.flock(stream.fileno(),fcntl.LOCK_UN)
        stream.close()


def atomic_new_bytes(path,raw):
    path=Path(path)
    if path.exists():raise FileExistsError('Immutable output already exists: '+str(path))
    with tempfile.NamedTemporaryFile(mode='wb',prefix='journal_',suffix='.tmp',dir=path.parent,delete=False) as f:
        f.write(raw);f.flush();os.fsync(f.fileno());temporary=Path(f.name)
    # Run lock prevents writer races. Windows rename also refuses an existing target.
    os.rename(temporary,path)


def atomic_new_json(path,value):
    atomic_new_bytes(path,json.dumps(value,ensure_ascii=False,allow_nan=False).encode('utf-8'))


class ObservationStore:
    def __init__(self,directory,plan,resume=False):
        self.directory=Path(directory)
        if resume:
            if not self.directory.is_dir() or load_json(self.directory/'plan.json')!=plan:raise ValueError('Resume input/code/device lineage differs')
        else:
            self.directory.mkdir(parents=True,exist_ok=False)
            write_json(self.directory/'plan.json',plan)
            (self.directory/'records').mkdir()
        self.plan=plan;self.plan_sha256=digest(self.directory/'plan.json')

    def path(self,id):
        return self.directory/'records'/(hashlib.sha256(id.encode()).hexdigest()+'.json')

    def read(self,identity):
        path=self.path(identity['image_id'])
        if not path.exists():return None
        item=load_json(path);proof=item.pop('payload_sha256',None)
        if proof!=content_hash(item):raise ValueError('Observation payload drift')
        if item['identity']!=identity or item['plan_sha256']!=self.plan_sha256:raise ValueError('Observation identity/input lineage mismatch')
        item['payload_sha256']=proof
        return item

    def append(self,identity,observation,rendering_context):
        value=dict(identity=identity,observation=observation,rendering_context=rendering_context,plan_sha256=self.plan_sha256)
        value['payload_sha256']=content_hash(value)
        atomic_new_json(self.path(identity['image_id']),value)

    def finish(self,identities):
        items=[self.read(i) for i in identities]
        if any(i is None for i in items):raise ValueError('Cannot seal partial inference coverage')
        # No directory discovery of dataset images. Own journal files are checked
        # only to reject unrecognized extras/duplicates from interrupted tooling.
        if len(list((self.directory/'records').glob('*.json')))!=len(items):raise ValueError('Unexpected journal records')
        raw=('\n'.join(json.dumps(i,ensure_ascii=False,allow_nan=False) for i in items)+'\n').encode('utf-8')
        path=self.directory/'observations.jsonl'
        if path.exists():
            if path.read_bytes()!=raw:raise ValueError('Existing consolidated raw observations differ')
        else:atomic_new_bytes(path,raw)
        completion=dict(complete=True,records=len(items),raw_sha256=digest(path),plan_sha256=self.plan_sha256,
                        dataset_manifest_sha256=self.plan['dataset_manifest_sha256'],selection_sha256=self.plan['selection_sha256'])
        marker=self.directory/'completion.json'
        if marker.exists():
            if load_json(marker)!=completion:raise ValueError('Completed run drift')
        else:atomic_new_json(marker,completion)
        return completion


def load_raw_run(directory,dataset):
    directory=Path(directory);completion=load_json(directory/'completion.json');plan=load_json(directory/'plan.json')
    if (not completion.get('complete') or completion['plan_sha256']!=digest(directory/'plan.json')
            or completion['raw_sha256']!=digest(directory/'observations.jsonl')):raise ValueError('Raw run incomplete or altered')
    if plan['dataset_manifest_sha256']!=dataset.manifest_sha256 or plan['selection_sha256']!=dataset.selection_sha256 or plan['split']!='development_design':
        raise ValueError('Raw run uses a different dataset/split')
    if plan['checkpoint_sha256']!=CHECKPOINT_SHA:raise ValueError('Raw run did not use the frozen E1 checkpoint')
    items=[json.loads(line) for line in (directory/'observations.jsonl').read_text(encoding='utf-8').splitlines()]
    expected={r['image_id']:dataset.identity(r) for r in dataset.rows}
    seen=set()
    for item in items:
        body={k:v for k,v in item.items() if k!='payload_sha256'}
        if content_hash(body)!=item['payload_sha256'] or item['plan_sha256']!=completion['plan_sha256']:raise ValueError('Raw observation payload/plan mismatch')
        id=item['identity']['image_id']
        if id in seen or id not in expected or item['identity']!=expected[id]:raise ValueError('Raw record duplication/selection mismatch')
        seen.add(id)
    if seen!=set(expected) or len(items)!=completion['records']:raise ValueError('Raw inference coverage incomplete')
    return items,dict(plan=plan,completion=completion,raw_directory=str(directory.resolve()))
