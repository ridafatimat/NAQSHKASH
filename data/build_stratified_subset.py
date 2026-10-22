"""Deterministic, parent-aware tier subset indices; no image/label contents read.

TRAIN and VALIDATION quantiles are derived independently from their full splits.
Duplicate quantiles, endpoints and cuts separating no observed values are merged.
Quantiles use linear interpolation at (n-1)*q; buckets are right-closed.
Bounded proportional (Hamilton) allocation keeps every nonempty stratum, with a
default floor of five samples (or its population). A one-sample floor is used if
the target cannot fund five per stratum; target < number of strata is rejected.
Within each stratum, all parents get a first opportunity before any second tier.
Global unseen parents are preferred, processing parent-scarce strata first.
This maximizes diversity within each stratum; global diversity is a greedy
objective, not a claim of a globally optimal combinatorial solution.
Repairs enforce occurrence floors, not equality of every character's frequency
ratio. Their additions are retained and distribution drift is reported.
Runtime is printed separately, keeping reports byte-reproducible.
Only new files under dataset_root/manifests may be written. No project imports.
"""
import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import time

VERSION = '1.0.0'
EXTRA_FIELDS = ['label_length_bucket', 'resized_width_bucket', 'stratum_id']


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def rank(seed, *parts):
    return hashlib.sha256(json.dumps([seed, *parts], ensure_ascii=False,
                                    separators=(',', ':')).encode('utf-8')).digest()


def quantile(sorted_values, q):
    pos = (len(sorted_values)-1)*q
    lo, hi = math.floor(pos), math.ceil(pos)
    return sorted_values[lo] + (sorted_values[hi]-sorted_values[lo])*(pos-lo)


def distribution(values):
    counts = Counter(values)
    ordered = sorted(counts)
    if not ordered:
        return None
    # Sorted observations also make the quantile convention explicit.
    expanded = sorted(values) if not isinstance(values, list) else sorted(values)
    return dict(count=len(expanded), min=ordered[0], max=ordered[-1],
                mean=sum(v*n for v,n in counts.items())/len(expanded),
                quantiles={str(q):quantile(expanded,q) for q in (.01,.05,.2,.25,.4,.5,.6,.75,.8,.9,.95,.99)},
                histogram={str(k):counts[k] for k in ordered})


def bucket_spec(values, requested):
    ordered = sorted(values)
    if not ordered or requested < 1:
        raise ValueError('Buckets require data and a positive bucket count')
    proposed = [quantile(ordered, i/requested) for i in range(1,requested)]
    cuts, partitions = [], set()
    # For discrete values, different numerical cuts can still induce the same
    # partition. Deduplicate by number of observations on the left, too.
    from bisect import bisect_right
    for cut in proposed:
        nleft = bisect_right(ordered, cut)
        if 0 < nleft < len(ordered) and nleft not in partitions:
            cuts.append(cut)
            partitions.add(nleft)
    return dict(requested_buckets=requested, actual_buckets=len(cuts)+1,
                proposed_quantile_boundaries=proposed, boundaries=cuts,
                min=ordered[0], max=ordered[-1], interval_rule='q1: x<=b1; qi: b(i-1)<x<=bi; last: x>b_last',
                method='linear interpolation at (n-1)*q, duplicate/empty partitions merged')


def bucket(value, spec, prefix):
    return prefix+'_q'+str(bisect_left(spec['boundaries'],value)+1)


def allocate(populations, target, minimum=5):
    keys = sorted(populations)
    total = sum(populations.values())
    target = min(target,total)
    if target < len(keys):
        raise ValueError('Target is too small to retain every nonempty stratum')
    lower = {k:min(minimum,populations[k]) for k in keys}
    if sum(lower.values()) > target:
        lower = {k:1 for k in keys}
    # Find bounded proportional quotas, retaining both floors and capacities.
    lo, hi = 0., 1.
    for _ in range(80):
        mid = (lo+hi)/2
        if sum(max(lower[k],min(populations[k],mid*populations[k])) for k in keys) < target:
            lo = mid
        else:
            hi = mid
    ideal = {k:max(lower[k],min(populations[k],hi*populations[k])) for k in keys}
    result = {k:min(populations[k],max(lower[k],math.floor(ideal[k]))) for k in keys}
    remainder = target-sum(result.values())
    eligible = sorted((k for k in keys if result[k]<populations[k]),
                      key=lambda k:(-(ideal[k]-result[k]),k))
    for k in eligible[:remainder]:
        result[k] += 1
    if remainder < 0 or sum(result.values()) != target:
        raise AssertionError('Allocation did not sum to target')
    return result,lower


def parent_sample(rows, strata, allocations, seed, split):
    groups, parent_ranks = {}, {}
    for sid, indices in strata.items():
        by_parent = defaultdict(list)
        for i in indices:
            by_parent[rows[i]['parent_image_id']].append(i)
        for parent, members in by_parent.items():
            members.sort(key=lambda i:rank(seed,split,'tier',rows[i]['tier_id']))
            parent_ranks.setdefault(parent,rank(seed,split,'parent',parent))
        groups[sid] = by_parent
    order = sorted(strata,key=lambda s:(len(groups[s]),s))
    remaining = dict(allocations)
    selected, parent_counts = set(), Counter()
    pass_number = 0
    while any(remaining.values()):
        changed = False
        for sid in order:
            if not remaining[sid]:
                continue
            eligible = [p for p,items in groups[sid].items() if len(items)>pass_number]
            eligible.sort(key=lambda p:(parent_counts[p],parent_ranks[p],p))
            for parent in eligible[:remaining[sid]]:
                selected.add(groups[sid][parent][pass_number])
                parent_counts[parent] += 1
                remaining[sid] -= 1
                changed = True
        if not changed:
            raise AssertionError('Parent sampling exhausted before meeting allocation')
        pass_number += 1
    audit = {}
    for sid, indices in strata.items():
        count = len({rows[i]['parent_image_id'] for i in indices if i in selected})
        maximum = min(allocations[sid],len(groups[sid]))
        assert count == maximum
        audit[sid] = dict(full_distinct_parents=len(groups[sid]),
                          initial_distinct_parents=count, maximum_at_allocation=maximum)
    return selected,audit


def character_counts(rows, indices):
    result = Counter()
    for i in indices:
        result.update(rows[i]['training_label'])
    return result


def repair(rows, selected, strata_sizes, vocabulary, minimum, seed, split):
    full = character_counts(rows,range(len(rows)))
    initial = character_counts(rows,selected)
    counts = initial.copy()
    thresholds = {c:min(minimum,full[c]) for c in vocabulary}
    deficient = {c for c in vocabulary if counts[c]<thresholds[c]}
    parents = Counter(rows[i]['parent_image_id'] for i in selected)
    sc = Counter(rows[i]['stratum_id'] for i in selected)
    additions = []
    inverted = defaultdict(list)
    symbols = {}
    if deficient:
        for i,row in enumerate(rows):
            if i not in selected:
                relevant = set(row['training_label']) & deficient
                if relevant:
                    symbols[i] = Counter(row['training_label'])
                    for char in sorted(relevant):
                        inverted[char].append(i)
    while deficient:
        char = min(deficient,key=lambda c:(full[c],c))
        candidates = [i for i in inverted[char] if i not in selected]
        if not candidates:
            raise AssertionError('Repair target is attainable but candidates exhausted')
        def score(i):
            row = rows[i]
            sid, parent = row['stratum_id'], row['parent_image_id']
            excess = max(0.,(sc[sid]+1)/(len(selected)+1)-strata_sizes[sid]/len(rows))
            gain = sum(min(symbols[i][c],thresholds[c]-counts[c]) for c in deficient)
            return (parent in parents,excess,-gain,parents[parent],rank(seed,split,'repair',row['tier_id']))
        chosen = min(candidates,key=score)
        before = {c:counts[c] for c in sorted(deficient) if symbols[chosen][c]}
        selected.add(chosen)
        counts.update(symbols[chosen])
        row = rows[chosen]
        additions.append(dict(tier_id=row['tier_id'], parent_image_id=row['parent_image_id'],
                              trigger_character=char, counts_before=before,
                              counts_added={c:symbols[chosen][c] for c in before}))
        parents[row['parent_image_id']] += 1
        sc[row['stratum_id']] += 1
        deficient = {c for c in deficient if counts[c]<thresholds[c]}
    full_total, subset_total = sum(full.values()),sum(counts.values())
    coverage = {c:dict(full_count=full[c],initial_subset_count=initial[c],subset_count=counts[c],
                      subset_full_ratio=counts[c]/full[c] if full[c] else None,
                      full_frequency_percent=100*full[c]/full_total,
                      subset_frequency_percent=100*counts[c]/subset_total,
                      normalized_frequency_ratio=(counts[c]/subset_total)/(full[c]/full_total) if full[c] else None,
                      effective_minimum=thresholds[c]) for c in vocabulary}
    return dict(characters=coverage, repair_samples_added=len(additions), repair_samples=additions,
                characters_repaired=sorted(c for c in vocabulary if initial[c]<thresholds[c]),
                unavailable_or_below_requested_minimum={c:full[c] for c in vocabulary if full[c]<minimum},
                final_minimum_observed_count=min(counts[c] for c in vocabulary))


def percentages(full, subset):
    nf, ns = sum(full.values()),sum(subset.values())
    return {k:dict(full_count=full[k],subset_count=subset[k],
                   full_percent=100*full[k]/nf,subset_percent=100*subset[k]/ns,
                   absolute_percentage_point_difference=abs(100*full[k]/nf-100*subset[k]/ns),
                   exceeds_2_percentage_points=abs(100*full[k]/nf-100*subset[k]/ns)>2)
            for k in sorted(full)}


def sample_split(rows, split, target, length_bins, width_bins, minimum, min_stratum, seed):
    length = bucket_spec([int(r['label_length']) for r in rows],length_bins)
    width = bucket_spec([int(r['resized_width']) for r in rows],width_bins)
    strata = defaultdict(list)
    for i,row in enumerate(rows):
        row['label_length_bucket'] = bucket(int(row['label_length']),length,'label')
        row['resized_width_bucket'] = bucket(int(row['resized_width']),width,'width')
        row['stratum_id'] = '+'.join((row['tier_name'],row['label_length_bucket'],row['resized_width_bucket']))
        strata[row['stratum_id']].append(i)
    populations = Counter({s:len(v) for s,v in strata.items()})
    allocation, floors = allocate(populations,target,min_stratum)
    selected,audit = parent_sample(rows,strata,allocation,seed,split)
    initial_indices = set(selected)
    return selected,initial_indices,strata,populations,allocation,floors,audit,length,width


def summarize_split(rows, selected, initial, strata, populations, allocation, floors, audit,
                    length, width, coverage, target):
    picked = [rows[i] for i in sorted(selected)]
    full_pc = Counter(r['parent_image_id'] for r in rows)
    subset_pc = Counter(r['parent_image_id'] for r in picked)
    initial_pc = {rows[i]['parent_image_id'] for i in initial}
    comparisons = {}
    for key in ('tier_name','label_length_bucket','resized_width_bucket','stratum_id'):
        comparisons[key] = percentages(Counter(r[key] for r in rows),Counter(r[key] for r in picked))
    strata_report = {}
    for sid, values in comparisons['stratum_id'].items():
        strata_report[sid] = dict(values, allocation_before_repairs=allocation[sid],
                                  allocation_floor=floors[sid], **audit[sid])
    return dict(full_sample_count=len(rows),target_sample_count=target,
                initial_sample_count=len(initial),final_sample_count=len(selected),
                full_distinct_parents=len(full_pc),initial_distinct_parents=len(initial_pc),
                subset_distinct_parents=len(subset_pc),parent_coverage_percent=100*len(subset_pc)/len(full_pc),
                bucket_boundaries=dict(label_length=length,resized_width=width),
                number_of_nonempty_strata=len(strata),
                stratum_sizes=dict(full_min=min(populations.values()),full_max=max(populations.values()),
                                  subset_min=min(v['subset_count'] for v in strata_report.values()),
                                  subset_max=max(v['subset_count'] for v in strata_report.values())),
                strata=strata_report,comparisons=comparisons,
                drift_flags=[dict(dimension=k,bucket=b,**v) for k,comp in comparisons.items()
                             for b,v in comp.items() if v['exceeds_2_percentage_points']],
                label_length_distribution=dict(full=distribution([int(r['label_length']) for r in rows]),
                                               subset=distribution([int(r['label_length']) for r in picked])),
                resized_width_distribution=dict(full=distribution([int(r['resized_width']) for r in rows]),
                                                subset=distribution([int(r['resized_width']) for r in picked])),
                tiers_per_parent=dict(full=distribution(list(full_pc.values())),
                                      subset_represented_parents=distribution(list(subset_pc.values())),
                                      subset_all_original_parents=distribution([subset_pc[p] for p in full_pc])),
                character_coverage=coverage)


def inventory(root, split):
    found = set()
    def visit(path):
        with os.scandir(path) as entries:
            for entry in entries:
                if entry.is_dir(follow_symlinks=False):
                    visit(Path(entry.path))
                elif entry.is_file():
                    found.add(Path(entry.path).relative_to(root).as_posix())
    for folder in ('images','labels'):
        visit(root/split/folder)
    return found


def valid_path(value, split, folder):
    path = PurePosixPath(value)
    return (bool(value) and not path.is_absolute() and '\\' not in value and ':' not in value
            and '..' not in path.parts and len(path.parts)>=3 and path.parts[:2]==(split,folder)
            and path.as_posix()==value)


def validate(rows, selected, split, vocabulary, files):
    issues = defaultdict(list)
    ids = Counter()
    for i in sorted(selected):
        row = rows[i]
        tid = row['tier_id']
        ids[tid] += 1
        if row['split'] != split: issues['wrong_split'].append(tid)
        for key,folder in (('image_path','images'),('label_path','labels')):
            if not valid_path(row[key],split,folder): issues['invalid_'+key].append(tid)
            elif row[key] not in files: issues['missing_'+folder].append(tid)
        target = row['training_label']
        required = len(target)+sum(a==b for a,b in zip(target,target[1:]))
        rw = max(1,round(int(row['image_width'])*(64/float(row['image_height']))))
        steps = max(1,rw//2//2)
        if not target: issues['empty_training_labels'].append(tid)
        if set(target)-vocabulary or row['unknown_characters']: issues['unknown_characters'].append(tid)
        if row['raw_label'].lstrip(' ') != target: issues['label_semantics_mismatch'].append(tid)
        if int(row['label_length']) != len(target): issues['label_length_mismatch'].append(tid)
        if int(row['leading_offset']) != len(row['raw_label'])-len(target): issues['leading_offset_mismatch'].append(tid)
        if int(row['resized_width']) != rw: issues['resized_width_mismatch'].append(tid)
        if required>steps or row['ctc_feasible']!='1': issues['ctc_infeasible'].append(tid)
        if int(row['ctc_required_timesteps'])!=required or int(row['crnn_output_length'])!=steps:
            issues['ctc_metadata_mismatch'].append(tid)
        if row['image_status']!='ok' or row['label_status']!='ok': issues['source_status_not_ok'].append(tid)
    duplicates = {k:v for k,v in ids.items() if v>1}
    keys = ['wrong_split','invalid_image_path','invalid_label_path','missing_images','missing_labels',
            'empty_training_labels','unknown_characters','label_semantics_mismatch','label_length_mismatch',
            'leading_offset_mismatch','resized_width_mismatch','ctc_infeasible','ctc_metadata_mismatch','source_status_not_ok']
    return dict(passed=not issues and not duplicates, duplicate_tier_ids=duplicates,
                counts={k:len(issues[k]) for k in keys},issue_ids=dict(issues))


def write_subset(path, rows, selected, fields):
    ordered = sorted(selected,key=lambda i:(rows[i]['split'],rows[i]['tier_id']))
    with path.open('x',encoding='utf-8',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=fields,quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows[i] for i in ordered)
    # Exact full-field CSV round-trip, not merely labels or selected IDs.
    count = 0
    with path.open(encoding='utf-8',newline='') as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == fields
        for i,row in zip(ordered,reader):
            assert row == {k:rows[i][k] for k in fields}
            count += 1
        assert next(reader,None) is None
    assert count == len(ordered)
    return dict(rows=count,all_fields_roundtrip_passed=True,sha256=sha256(path))


def markdown_report(report):
    out = ['# NAQSHKASH stratification report','',
           'Manifest/index generation only. Full TEST remains the sole final test set.','',
           '| Split | Full tiers | Final tiers | Full parents | Subset parents | Parent coverage | Repair additions |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for split in ('train','validation'):
        r = report['splits'][split]
        out.append('| {} | {} | {} | {} | {} | {:.4f}% | {} |'.format(split,r['full_sample_count'],r['final_sample_count'],r['full_distinct_parents'],r['subset_distinct_parents'],r['parent_coverage_percent'],r['character_coverage']['repair_samples_added']))
    for split in ('train','validation'):
        r = report['splits'][split]
        out += ['', '## '+split.upper(), '', '| Dimension | Bucket | Full % | Subset % | Absolute pp difference | >2 pp |', '|---|---|---:|---:|---:|---|']
        for dim in ('tier_name','label_length_bucket','resized_width_bucket'):
            for b,v in r['comparisons'][dim].items():
                out.append('| {} | {} | {:.4f} | {:.4f} | {:.4f} | {} |'.format(dim,b,v['full_percent'],v['subset_percent'],v['absolute_percentage_point_difference'],v['exceeds_2_percentage_points']))
        out += ['', '| Measurement | Set | Min | P25 | Median | P75 | P95 | P99 | Max |', '|---|---|---:|---:|---:|---:|---:|---:|---:|']
        for dim in ('label_length_distribution','resized_width_distribution','tiers_per_parent'):
            for name,stats in r[dim].items():
                out.append('| {} | {} | {} | {} | {} | {} | {} | {} | {} |'.format(dim,name,stats['min'],*(stats['quantiles'][q] for q in ('0.25','0.5','0.75','0.95','0.99')),stats['max']))
        out += ['', '| Character (JSON escaped) | Full occurrences | Subset occurrences | Subset/full | Normalized frequency ratio |', '|---|---:|---:|---:|---:|']
        for c,v in r['character_coverage']['characters'].items():
            out.append('| {} | {} | {} | {} | {} |'.format(json.dumps(c,ensure_ascii=False).replace('|','&#124;'),v['full_count'],v['subset_count'],v['subset_full_ratio'],v['normalized_frequency_ratio']))
        out += ['', 'Bucket boundaries: `'+json.dumps(r['bucket_boundaries'],ensure_ascii=False)+'`', '',
                'Characters repaired: `'+json.dumps(r['character_coverage']['characters_repaired'],ensure_ascii=False)+'`']
    out += ['', '## Integrity and reproducibility','', '```json',json.dumps({k:report[k] for k in ('settings','leakage','integrity','reproducibility')},ensure_ascii=False,indent=2,sort_keys=True),'```','',
            'Full stratum counts, character counts, histograms, repairs, and assumptions are in stratification_report.json. Runtime is printed by the command and omitted here for byte-reproducibility.','']
    return '\n'.join(out)


def build(manifest, output_dir, train_target=150000, val_target=10000, min_char_count=50,
          num_length_buckets=5, num_width_buckets=5, seed=42, min_stratum_count=5,
          dataset_root=None, summary_path=None):
    started = time.perf_counter()
    manifest = Path(manifest).resolve()
    root = Path(dataset_root).resolve() if dataset_root else manifest.parent.parent
    output = Path(output_dir).resolve()
    allowed = (root/'manifests').resolve()
    if output!=allowed and allowed not in output.parents:
        raise ValueError('Output must be within dataset_root/manifests')
    if min(train_target,val_target,num_length_buckets,num_width_buckets,min_stratum_count)<1 or min_char_count<0:
        raise ValueError('Targets/bucket counts/floors must be positive; character minimum must be nonnegative')
    names = ['stratified_train_150k.csv','stratified_validation_10k.csv','stratification_report.json','STRATIFICATION_REPORT.md']
    if any((output/n).exists() for n in names):
        raise FileExistsError('Outputs already exist; choose a new --output-dir (no overwrite)')
    summary_path = Path(summary_path) if summary_path else manifest.parent/'manifest_summary.json'
    input_hash, summary_hash = sha256(manifest),sha256(summary_path)
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    if summary['manifest_sha256']!=input_hash:
        raise ValueError('Input manifest SHA256 differs from its validated summary')
    vocabulary = sorted(summary['vocabulary'])
    if len(vocabulary)!=65 or summary['vocabulary_classes_including_blank']!=66:
        raise ValueError('Expected existing 65 symbols plus blank; vocabulary will not be changed')
    print('Reading validated full manifest...',flush=True)
    rows = {'train':[],'validation':[]}
    parents = {s:set() for s in ('train','validation','test')}
    ids = {s:set() for s in parents}
    with manifest.open(encoding='utf-8',newline='') as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        if any(k in fields for k in EXTRA_FIELDS):
            raise ValueError('Input must be the full manifest without subset annotations')
        for row in reader:
            split = row['split']
            if split not in parents or not row['parent_image_id']:
                raise ValueError('Unexpected split or missing explicit parent ID')
            if row['tier_id'] in ids[split]:
                raise ValueError('Duplicate input tier ID: '+row['tier_id'])
            ids[split].add(row['tier_id'])
            parents[split].add(row['parent_image_id'])
            if split in rows:
                rows[split].append(row)
    if set(vocabulary)!=set(character_counts(rows['train'],range(len(rows['train'])))):
        raise ValueError('Actual full TRAIN vocabulary differs from validated vocabulary')
    report = dict(script_version=VERSION,settings=dict(seed=seed,train_target=train_target,val_target=val_target,
                  min_char_count=min_char_count,num_length_buckets=num_length_buckets,
                  num_width_buckets=num_width_buckets,min_stratum_count=min_stratum_count),
                  columns=fields+EXTRA_FIELDS,splits={},integrity={},leakage={},outputs={},
                  reproducibility=dict(input_manifest_sha256=input_hash,input_summary_sha256=summary_hash,
                                       script_sha256=sha256(__file__),selection_hash='SHA256(seed, split, purpose, explicit ID)',
                                       output_order='split, tier_id',runtime='printed separately; excluded from deterministic report'),
                  assumptions=__doc__,test_policy='No test subset; TEST read only for parent/ID integrity and leakage checks.')
    chosen_parents = {}
    chosen_ids = {}
    selections = {}
    for split,target in (('train',train_target),('validation',val_target)):
        rows[split].sort(key=lambda r:r['tier_id'])
        print('{}: stratifying {} rows, target {}'.format(split,len(rows[split]),target),flush=True)
        data = sample_split(rows[split],split,target,num_length_buckets,num_width_buckets,min_char_count,min_stratum_count,seed)
        selected,initial,strata,populations,allocation,floors,audit,length,width = data
        coverage = repair(rows[split],selected,populations,vocabulary,min_char_count,seed,split)
        print('{}: {} selected (+{} repairs); checking file inventory...'.format(split,len(selected),coverage['repair_samples_added']),flush=True)
        validation = validate(rows[split],selected,split,set(vocabulary),inventory(root,split))
        if not validation['passed']:
            raise ValueError('Selected sample validation failed: '+json.dumps(validation))
        report['integrity'][split] = validation
        report['splits'][split] = summarize_split(rows[split],selected,initial,strata,populations,allocation,floors,audit,length,width,coverage,target)
        chosen_parents[split] = {rows[split][i]['parent_image_id'] for i in selected}
        chosen_ids[split] = {rows[split][i]['tier_id'] for i in selected}
        selections[split] = selected
    for a,b in (('train','validation'),('train','test'),('validation','train'),('validation','test')):
        overlap = sorted(chosen_parents[a]&parents[b])
        report['leakage']['selected_'+a+'_vs_full_'+b] = dict(parent_overlap_count=len(overlap),parent_ids=overlap,
                                                            tier_id_overlap_count=len(chosen_ids[a]&ids[b]))
    report['leakage']['selected_train_vs_selected_validation'] = dict(parent_overlap_count=len(chosen_parents['train']&chosen_parents['validation']),
                                                                       tier_id_overlap_count=len(chosen_ids['train']&chosen_ids['validation']))
    if any(v['parent_overlap_count'] or v['tier_id_overlap_count'] for v in report['leakage'].values()):
        raise ValueError('Leakage detected; no subset outputs written')
    if sha256(manifest)!=input_hash or sha256(summary_path)!=summary_hash:
        raise ValueError('Protected input changed during processing')
    output.mkdir(parents=True,exist_ok=True)
    for split,name in (('train',names[0]),('validation',names[1])):
        report['outputs'][split] = dict(filename=name,**write_subset(output/name,rows[split],selections[split],fields+EXTRA_FIELDS))
    with (output/names[2]).open('x',encoding='utf-8',newline='\n') as f:
        f.write(json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
    with (output/names[3]).open('x',encoding='utf-8',newline='\n') as f:
        f.write(markdown_report(report))
    print(json.dumps({s:{k:report['splits'][s][k] for k in ('final_sample_count','subset_distinct_parents','parent_coverage_percent','stratum_sizes','number_of_nonempty_strata','drift_flags')} for s in rows},indent=2),flush=True)
    print('RUNTIME_SECONDS={:.3f}'.format(time.perf_counter()-started),flush=True)
    return report


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,default=Path('datasets/Final tier dataset/manifests/full_tier_manifest.csv'))
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--dataset-root',type=Path)
    parser.add_argument('--summary',type=Path)
    parser.add_argument('--train-target',type=int,default=150000)
    parser.add_argument('--val-target',type=int,default=10000)
    parser.add_argument('--min-char-count',type=int,default=50)
    parser.add_argument('--num-length-buckets',type=int,default=5)
    parser.add_argument('--num-width-buckets',type=int,default=5)
    parser.add_argument('--min-stratum-count',type=int,default=5)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--target-small',action='store_true',help='Use full-source boundaries with 10,000 TRAIN and 2,000 VALIDATION targets')
    args = parser.parse_args()
    if args.target_small:
        args.train_target,args.val_target = 10000,2000
    output = args.output_dir or args.manifest.parent/('stratification_small' if args.target_small else '')
    build(args.manifest,output,args.train_target,args.val_target,args.min_char_count,
          args.num_length_buckets,args.num_width_buckets,args.seed,args.min_stratum_count,args.dataset_root,args.summary)
