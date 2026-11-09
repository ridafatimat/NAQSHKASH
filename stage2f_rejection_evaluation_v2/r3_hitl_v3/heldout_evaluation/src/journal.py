"""Exclusive new inference records, exact identity and release binding."""
from pathlib import Path
from release_io import load,sha,canonical,write_json,output_path
from heldout_dataset import identity


class Journal:
    def __init__(self,directory,plan,resume=False):
        self.directory=output_path(directory)
        if resume:
            if load(self.directory/'plan.json')!=plan:raise ValueError('Resume plan/code/selection mismatch')
        else:
            self.directory.mkdir(parents=True,exist_ok=False);write_json(self.directory/'plan.json',plan);(self.directory/'records').mkdir()
        self.plan=plan;self.plan_sha256=sha(self.directory/'plan.json')

    def path(self,id):
        import hashlib
        return self.directory/'records'/(hashlib.sha256(id.encode()).hexdigest()+'.json')

    def read(self,item_identity):
        p=self.path(item_identity['image_id'])
        if not p.exists():return None
        r=load(p)
        if r['identity']!=item_identity or r['plan_sha256']!=self.plan_sha256 or r['payload_sha256']!=canonical({k:v for k,v in r.items() if k!='payload_sha256'}):raise ValueError('Journal identity/payload drift')
        return r

    def append(self,item_identity,observation,context,geometry):
        from observation_store import atomic_new_json
        r=dict(identity=item_identity,observation=observation,rendering_context=context,geometry=geometry,plan_sha256=self.plan_sha256)
        r['payload_sha256']=canonical(r);atomic_new_json(self.path(item_identity['image_id']),r)

    def finish(self,dataset):
        rows=[self.read(identity(r)) for r in dataset.rows]
        if any(r is None for r in rows) or len(list((self.directory/'records').glob('*.json')))!=len(rows):raise ValueError('Raw observation coverage incomplete')
        path=self.directory/'observations.jsonl'
        import json
        raw=('\n'.join(json.dumps(r,ensure_ascii=False,allow_nan=False) for r in rows)+'\n').encode('utf-8')
        if not path.exists():
            from observation_store import atomic_new_bytes
            atomic_new_bytes(path,raw)
        else:
            if path.read_bytes()!=raw:raise ValueError('Existing consolidated raw output differs')
        done=dict(complete=True,kind='raw',split=dataset.split,records=len(rows),parents=len(dataset.groups),release_sha256=self.plan['release_sha256'],
                  outputs={n:sha(self.directory/n) for n in ('plan.json','observations.jsonl')})
        if (self.directory/'completion.json').exists():
            if load(self.directory/'completion.json')!=done:raise ValueError('Completed inference marker differs')
        else:
            from observation_store import atomic_new_json
            atomic_new_json(self.directory/'completion.json',done)
        return done


def read_completed(directory,dataset,release_sha256):
    from release_io import check_completion
    done=check_completion(directory,'raw');plan=load(Path(directory)/'plan.json')
    if done['split']!=dataset.split or done['records']!=600 or done['parents']!=150 or plan['split']!=dataset.split or plan['selection_sha256']!=dataset.selection_sha256 or plan['dataset_manifest_sha256']!=dataset.manifest_sha256 or plan['release_sha256']!=release_sha256:
        raise ValueError('Raw observations do not match exact authorized held-out split/release')
    import json
    rows=[json.loads(line) for line in (Path(directory)/'observations.jsonl').read_text(encoding='utf-8').splitlines()]
    if len(rows)!=600 or len({r['identity']['image_id'] for r in rows})!=600 or {r['identity']['image_id'] for r in rows}!=set(dataset.by_id):raise ValueError('Raw IDs are not selected held-out coverage')
    plan_hash=sha(Path(directory)/'plan.json')
    for r in rows:
        if r['identity']!=identity(dataset.by_id[r['identity']['image_id']]) or r['plan_sha256']!=plan_hash or r['payload_sha256']!=canonical({k:v for k,v in r.items() if k!='payload_sha256'}):raise ValueError('Raw identity or payload proof differs')
    return rows,plan
