"""Read-only final release integrity and approved saved-decision preflight."""
import argparse,hashlib,json,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def load(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def verify(root=ROOT,adapter=True):
    root=Path(root).resolve();path=root/'release_manifest.json'
    if sha(path)!=(root/'RELEASE_SHA256.txt').read_text().strip():raise ValueError('Release manifest seal mismatch')
    m=load(path)
    if m['policy_id']!='R3_HITL_V3' or m['status']!='FINAL_FROZEN':raise ValueError('Wrong release')
    for relative,entry in m['files'].items():
        p=(root/relative).resolve()
        if not p.is_relative_to(root) or sha(p)!=entry['sha256']:raise ValueError('Frozen artifact drift: '+relative)
    if adapter:
        folder=root.parent/'heldout_evaluation'
        for relative,h in m['heldout_adapter_sha256'].items():
            p=(folder/relative).resolve()
            if not p.is_relative_to(folder) or sha(p)!=h:raise ValueError('Frozen held-out adapter drift: '+relative)
    return m


def replay(root=ROOT):
    from freeze_runtime import FrozenRuntime
    runtime=FrozenRuntime(root);counts={v:{o:0 for o in ('ACCEPT','REVIEW_UNCERTAIN','REJECT_INVALID')} for v in ('VALID','INVALID')}
    parents=set();checked=0
    with (Path(root)/'provenance/development_replay.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            item=json.loads(line)
            result=runtime.policy.decide(item['R2'],item['text'],item['availability'],runtime.configuration,item['alignment_evidence'])
            if canonical(result['decision'])!=item['expected_decision_sha256'] or canonical(result['findings'])!=item['expected_findings_sha256']:
                raise ValueError('Approved saved-decision reproduction failed: '+item['image_id'])
            if item['R2']['outcome']=='REVIEW_UNCERTAIN' and result['decision']['outcome']!='REVIEW_UNCERTAIN':raise ValueError('R2 REVIEW promoted')
            counts[item['offline_validity']][result['decision']['outcome']]+=1;parents.add(item['parent']);checked+=1
    expected=runtime.manifest['approved_development']['three_way_confusion']
    if checked!=2800 or len(parents)!=700 or counts!=expected:raise ValueError('Approved development replay coverage/count mismatch')
    pipeline_checks=0
    for line in (Path(root)/'provenance/post_OCR_pipeline_smoke.jsonl').read_text(encoding='utf-8').splitlines():
        item=json.loads(line);result=runtime.decide(item['observation'],item['rendering_context'])
        if canonical(result['R2'])!=item['expected_R2_sha256'] or canonical(result['R3']['decision'])!=item['expected_R3_sha256']:raise ValueError('Frozen post-OCR pipeline mismatch: '+item['image_id'])
        pipeline_checks+=1
    return dict(saved_decisions_reproduced=checked,post_OCR_pipeline_smoke_cases=pipeline_checks,parents=len(parents),counts=counts,pixel_reads=0,OCR_inference=False,
                scope='Integrity preflight of approved cached R3 evidence, not a new OCR/development experiment',local_modules=runtime.loaded_sources())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--replay-approved-development',action='store_true');a=p.parse_args()
    m=verify();result=dict(verified=True,policy_id=m['policy_id'],release_sha256=sha(ROOT/'release_manifest.json'),frozen_files=len(m['files']),existing_snapshots_verified=m['existing_snapshot_count'])
    if a.replay_approved_development:result['reproduction']=replay()
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
