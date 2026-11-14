"""Semantic source-overlap index. Full historical sweep is MANUAL, never implicit."""
import argparse
from collections import Counter
from common_v2 import *


def bounded_references():
    refs={s['name']:set() for s in source_specs()}
    names={'normal':'stage1_100000','general':'general_10000','positive':'positive_edges'}
    for r in read_csv(PROJECT/'stage2/preprocessing_improvements/manifests/parents.csv'):
        spec=next(s for s in source_specs() if s['name']==names[r['dataset']])
        refs[spec['name']].add(spec['prefix']+r['source_json'])
    for name in ('evaluation.csv','synthetic_replay_train.csv'):
        for r in read_csv(PROJECT/'stage2/stage2e_finetuning/manifests'/name):
            pid=r['parent_id']
            if pid.startswith(('train_','validation_','test_')):
                refs['stage1_100000'].add('datasets/talim_100000/'+pid.split('_')[0]+'/json/'+pid+'.json')
    # All326 edge-parent identities, including historical protected ones, are
    # read only for overlap hashes, never as generation/rule-design examples.
    refs['positive_edges']={'talim_edge_cases_positive_v1/json/edge_'+str(i).zfill(5)+'.json' for i in range(1,327)}
    return refs


def build_index(mode):
    refs=bounded_references() if mode=='bounded' else None
    keys=set();counts={};errors=[];fingerprints={}
    for spec in source_specs():
        with zipfile.ZipFile(spec['path']) as z:
            members=sorted(refs[spec['name']]) if refs else sorted(n for n in z.namelist() if n.startswith(spec['prefix']) and '/json/' in n and n.endswith('.json'))
            counts[spec['name']]=len(members)
            for member in members:
                try:keys.add(rules.semantic_key(json.loads(z.read(member))))
                except (ValueError,KeyError,TypeError) as exc:errors.append(dict(archive=spec['name'],member=member,error=str(exc)))
            fingerprints[spec['name']]=dict(path=spec['path'],size=Path(spec['path']).stat().st_size,
                sha256=digest(spec['path']) if mode=='full' else None,
                verification='full archive hash' if mode=='full' else 'bounded JSON metadata lookup; full old dataset not swept')
    old=DATA/'talim_edge_cases_negative_v1'
    for path in (old/'actual_symbols/json/negative.json',old/'duplicate_id/json/second.json'):
        keys.add(rules.semantic_key(read_json(path)))
    complete=mode=='full' and not errors and counts=={'stage1_100000':100000,'general_10000':10000,'positive_edges':326}
    return dict(schema_version=1,mode=mode,full_historical_coverage=complete,semantic_keys=sorted(keys),
        source_record_counts=counts,errors=errors,archives=fingerprints,
        old_negative_families_indexed=1,old_negative_content_templates_indexed=2,
        meaning='source semantic identity overlap, not a new model evaluation; old test annotations used only to avoid overlap',
        limitation='archive scope covers known Stage1/2C/2D/2E/2F canonical sources; no claim about unknown external data or perceptual near-duplicates')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('bounded','full'),required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if not args.output.resolve().is_relative_to(ROOT.resolve()):raise ValueError('Index output must stay in dataset-v2 workspace')
    if args.output.exists():raise FileExistsError(args.output)
    result=build_index(args.mode);write_json(args.output,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('semantic_keys','archives')},indent=2))


if __name__=='__main__':main()
