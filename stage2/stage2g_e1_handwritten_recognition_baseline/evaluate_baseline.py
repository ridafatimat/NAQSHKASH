"""Recognition-only E1 evaluation. Selective validation extraction; no segmentation."""
import sys,csv,json,hashlib,io,re,zipfile,unittest,collections
from pathlib import Path
import cv2,numpy as np
from PIL import Image
sys.dont_write_bytecode=True
PROJECT=Path(r'C:\NAQSHKASH')
OUT=PROJECT/'stage2/stage2g_e1_handwritten_recognition_baseline'
ARCHIVE=Path(r'C:\Users\hp\Downloads\handwritten_tier_crops (5).zip')
IDS={'all_003191','all_003556','all_009197'}
sys.path[:0]=[str(PROJECT),str(PROJECT/'stage2/stage2e_finetuning/src')]
import context as ctx
import evaluation as stagee
import metrics as em
def sha(b):return hashlib.sha256(b).hexdigest()
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def csvout(p,rows,fields=None):
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader();w.writerows(rows)
def no_space(s):return re.sub(r'\s+','',s)
def edit_events(a,b):
    dp=[[0]*(len(b)+1) for _ in range(len(a)+1)]
    for i in range(len(a)+1):dp[i][0]=i
    for j in range(len(b)+1):dp[0][j]=j
    for i in range(1,len(a)+1):
        for j in range(1,len(b)+1):dp[i][j]=dp[i-1][j-1] if a[i-1]==b[j-1] else min(dp[i-1][j-1],dp[i-1][j],dp[i][j-1])+1
    result=[];i=len(a);j=len(b)
    while i or j:
        if i and j and a[i-1]==b[j-1]:i-=1;j-=1
        elif i and j and dp[i][j]==dp[i-1][j-1]+1:result.append(('substitution',a[i-1],b[j-1]));i-=1;j-=1
        elif i and dp[i][j]==dp[i-1][j]+1:result.append(('deletion',a[i-1],''));i-=1
        else:result.append(('insertion','',b[j-1]));j-=1
    return result
def aggregate(rows):
    if not rows:return {'tiers':0,'cer':None,'exact_accuracy':None}
    m=em.aggregate(rows);nos={k:sum(r['no_space_'+k] for r in rows) for k in ['substitutions','deletions','insertions']}
    chars=sum(len(no_space(r['truth'])) for r in rows)
    return dict(m,cer_including_spaces=m['cer'],cer_excluding_spaces=sum(nos.values())/chars if chars else None,no_space_characters=chars,no_space_edit_counts=nos,exact_matches=sum(r['exact'] for r in rows))
def main():
    OUT.mkdir(exist_ok=True)
    if (OUT/'metrics.json').exists():raise FileExistsError('Preserve existing completed results')
    data=OUT/'inputs';data.mkdir(exist_ok=False);snap={str(ARCHIVE):sha(ARCHIVE.read_bytes())}
    with zipfile.ZipFile(ARCHIVE) as z:
        manifests=[n for n in z.namelist() if n.endswith('/validation_manifest.csv')];assert len(manifests)==1
        prefix=manifests[0][:-len('validation_manifest.csv')]
        wanted=['validation_manifest.csv','manifest.csv','cropping_report.md','scripts/crop.py','scripts/fix_validation.py','scripts/final_row01_rewrite.py']
        manifest=list(csv.DictReader(io.StringIO(z.read(manifests[0]).decode('utf-8-sig'),newline='')))
        for r in manifest:wanted.extend([r['crop_path'],r['label_path']])
        wanted+= [n[len(prefix):] for n in z.namelist() if n.startswith(prefix+'validation/source_photos/') and not n.endswith('/')]
        # Never open/read/extract protected-test image or label members.
        for rel in sorted(set(wanted)):
            assert not Path(rel).is_absolute() and '..' not in Path(rel).parts and 'protected_test' not in Path(rel).parts
            if rel not in ['validation_manifest.csv','manifest.csv','cropping_report.md'] and not rel.startswith(('validation/','scripts/')):raise ValueError('non-validation crop extraction')
            b=z.read(prefix+rel);dest=data/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(b);snap[str(dest)]=sha(b)
    manifest=read(data/'validation_manifest.csv');development=read(data/'manifest.csv')
    training_ready=sum(r['split']=='train' and r['training_ready'].lower()=='true' for r in development)
    training_pending=sum(r['split']=='train' and r['training_ready'].lower()!='true' for r in development)
    release_path=PROJECT/'stage2/stage2e_finetuning/reports/selected_checkpoint.json';release=json.loads(release_path.read_text());assert release['selected']=='E1'
    checkpoint=Path(release['selected_checkpoint']);assert sha(checkpoint.read_bytes())==release['sha256'];snap[str(checkpoint)]=release['sha256'];snap[str(release_path)]=sha(release_path.read_bytes())
    split=read(PROJECT/'stage2/stage2g_handwritten_dataset_preparation_20261108/source_split.csv');validation={r['source_id'] for r in split if r['split']=='validation'};assert validation==IDS
    assert len(manifest)==14 and len({r['tier_id'] for r in manifest})==14 and {r['source_id'] for r in manifest}==IDS
    ctx.seed();model,vocab,state=ctx.model_load(checkpoint);model.eval()
    prepared=[];errors=[];pixels=[];validation_checks=[]
    annotations={s:json.loads((PROJECT/'stage2/stage2g_handwritten_dataset_preparation_20261108/annotations'/(s+'.json')).read_text()) for s in IDS}
    for r in manifest:
        try:
            assert r['split']=='validation' and r['usage']=='VALIDATION_READY' and r['status']=='CONFIDENT'
            assert r['tier'] in ['upper','count','lower'] and r['source_id'] in validation
            assert r['crop_path']==f"validation/tier_crops/{r['tier_id']}.png" and r['label_path']==f"validation/tier_labels/{r['tier_id']}.txt"
            assert r['tier_id']==f"{r['source_id']}_row_{int(r['row_index']):02d}_{r['tier']}"
            path=data/r['crop_path'];b=path.read_bytes()
            with Image.open(io.BytesIO(b)) as im:im.verify()
            im=cv2.imread(str(path),0);assert im is not None and im.size and int(im.max())>int(im.min())
            assert im.shape==(int(r['crop_height']),int(r['crop_width']))
            if r['crop_sha256']:assert sha(b)==r['crop_sha256']
            raw=(data/r['label_path']).read_text(encoding='utf-8-sig');assert raw==r['raw_label']
            key='count_line' if r['tier']=='count' else r['tier'];assert raw==annotations[r['source_id']]['logical_rows'][int(r['row_index'])][key]
            truth=raw.lstrip(' ');assert truth==r['ocr_text'] and len(raw)-len(truth)==int(r['leading_offset'])
            assert vocab.decode(vocab.encode(truth))==truth
            prepared.append(ctx.row_from_image(path,truth,id=r['tier_id'],cohort='handwritten_validation',parent_id=r['source_id'],case_id=r['source_id'],kind='manually_prepared_handwriting',augmentation='none',role=r['tier']))
            pixels.append({'tier_id':r['tier_id'],'nonwhite_pixels':int(np.sum(im<255)),'dark_pixels_le_200':int(np.sum(im<=200)),'light_nonwhite_pixels_201_254':int(np.sum((im>200)&(im<255))),'white_fraction':float(np.mean(im==255))})
            validation_checks.append({'tier_id':r['tier_id'],'mapping':'PASS','crop_sha256':sha(b),'label_sha256':sha((data/r['label_path']).read_bytes()),'auto_flags':r['auto_flags'],'quality_note':r['quality_note'],'independent_talim_verification':False})
        except Exception as exc:errors.append({'tier_id':r['tier_id'],'stage':'input_validation','error':repr(exc)})
    predictions=[]
    for r in prepared:
        try:
            _,pred=stagee.evaluate(model,vocab,[r],output=None,device='cpu');p=pred[0]
            p['source_id']=r['parent_id'];mr=next(x for x in manifest if x['tier_id']==r['id']);p['row_index']=mr['row_index'];p['raw_reference']=mr['raw_label']
            counts=ctx.dc.old.edit_counts(no_space(p['truth']),no_space(p['prediction']))
            p.update({'no_space_'+k:v for k,v in counts.items()});p['status']='PROVISIONAL';p['qualification']='Current corrected manifest mapping checked; visual tier QA, not independent Talim-reader verification'
            predictions.append(p)
        except Exception as exc:errors.append({'tier_id':r['id'],'stage':'inference','error':repr(exc)})
    assert predictions,'No inference successes'
    overall=aggregate(predictions);per_source={s:aggregate([r for r in predictions if r['source_id']==s]) for s in sorted(IDS)}
    events=collections.Counter(e for p in predictions for e in edit_events(p['truth'],p['prediction']))
    event_counts={k:sum(n for (op,a,b),n in events.items() if op==k) for k in ['substitution','deletion','insertion']}
    assert event_counts==dict(substitution=overall['substitutions'],deletion=overall['deletions'],insertion=overall['insertions'])
    rewrite=data/'validation/source_photos/all_003191_row01_rewrite_original.jpg';rewrite_hash=sha(rewrite.read_bytes());assert rewrite_hash.startswith('7af54bc69e92490f')
    for r in manifest:
        if r['source_id']=='all_003191' and r['row_index']=='1':assert r['correction_date']=='2026-11-09' and r['crop_sha256'] and r['crop_sha256']!=r['previous_crop_sha256']
    metrics={'status':'PROVISIONAL','evaluation_type':'recognition-only on supplied manually prepared/corrected validation crops; NOT end-to-end page OCR','dataset_archive':str(ARCHIVE),'dataset_archive_sha256':snap[str(ARCHIVE)],'checkpoint':str(checkpoint),'checkpoint_release':'E1','checkpoint_sha256':release['sha256'],'vocabulary_hash':state['vocabulary_hash'],'expected_tiers':14,'scored_tiers':len(predictions),'scored_sources':len({r['source_id'] for r in predictions}),'input_exclusions':sum(e['stage']=='input_validation' for e in errors),'inference_failures':sum(e['stage']=='inference' for e in errors),'overall':overall,'per_source':per_source,'per_role':{role:aggregate([p for p in predictions if p['role']==role]) for role in ['upper','count','lower']},'most_common_character_errors':[{'operation':op,'reference':a,'prediction':b,'count':n} for (op,a,b),n in events.most_common(15)],'validation_checks':validation_checks,'pixel_diagnostics':pixels,'training_manifest_metadata_only':{'training_ready':training_ready,'training_pending':training_pending},'corrected_rephotograph':{'sha256':rewrite_hash,'row':'all_003191 row 1','current_crop_hashes_match_manifest':True,'stale_warnings':'older report text, auto_flags and observed_ink_groups retained; current corrected crops and later report update take precedence','semantic_verification':'visual evidence supports five upper marks; independent Talim reader verification incomplete'},'faint_ink_limitation':'Archived crop.py render_tier explicitly whitens all retained grayscale pixels >200. Corrected all_003556 lower uses a keep-faint exception. Exact current erased-ink percentage cannot be recovered from final PNGs without original masks. Earlier report estimate of seven affected validation tiers predates latest rephotograph. No re-render or diagnostic inference performed.'}
    csvout(OUT/'predictions.csv',predictions);csvout(OUT/'inference_errors.csv',errors,['tier_id','stage','error'])
    # Protect inputs/checkpoint byte-for-byte after inference.
    for p,h in snap.items():assert sha(Path(p).read_bytes())==h,p
    class ValidationTests(unittest.TestCase):
        def test_validation_coverage(self):self.assertEqual(len(prepared),14);self.assertEqual(len(predictions),14);self.assertEqual({p['source_id'] for p in predictions},IDS)
        def test_mapping_and_hashes(self):self.assertEqual(len(validation_checks),14);self.assertTrue(all(x['mapping']=='PASS' for x in validation_checks))
        def test_split_isolation(self):self.assertFalse(any('protected_test' in p.parts for p in data.rglob('*')));self.assertEqual(validation,IDS)
        def test_frozen_e1_and_vocabulary(self):self.assertEqual(sha(checkpoint.read_bytes()),release['sha256']);self.assertEqual(state['char_to_idx'],vocab.char_to_index)
        def test_metrics_and_whitespace(self):
            self.assertEqual(sum(overall[k] for k in ['substitutions','deletions','insertions'])/overall['characters'],overall['cer_including_spaces']);self.assertEqual(no_space(' A\t B\n'),'AB');self.assertEqual(ctx.dc.old.edit_counts('AB','AC'),{'substitutions':1,'deletions':0,'insertions':0})
        def test_rephotograph_current(self):self.assertTrue(rewrite_hash.startswith('7af54bc69e92490f'));self.assertTrue(all(x['correction_date']=='2026-11-09' for x in manifest if x['source_id']=='all_003191' and x['row_index']=='1'))
        def test_no_failures_and_training_metadata(self):self.assertEqual(errors,[]);self.assertEqual((training_ready,training_pending),(88,81))
    with (OUT/'validation_test_results.txt').open('w',encoding='utf-8') as stream:result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ValidationTests))
    metrics['validation_tests']={'passed':result.testsRun-len(result.failures)-len(result.errors),'failed':len(result.failures)+len(result.errors)}
    (OUT/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    report=f'''# Frozen E1 handwritten recognition baseline — PROVISIONAL

Recognition-only evaluation of the supplied corrected crops, not end-to-end page OCR. No page preprocessing, segmentation, tier detection, training or test-set evaluation was run.

## Actual results
| Metric | Result |
|---|---:|
| Sources scored | {metrics['scored_sources']} |
| Tiers scored | {len(predictions)}/14 |
| CER including spaces | {overall['cer_including_spaces']:.6f} ({overall['cer_including_spaces']:.2%}) |
| CER excluding all whitespace | {overall['cer_excluding_spaces']:.6f} ({overall['cer_excluding_spaces']:.2%}) |
| Exact-tier accuracy (Stage 2E) | {overall['exact_accuracy']:.2%} ({overall['exact_matches']}/{len(predictions)}) |
| Substitutions / deletions / insertions (spaces included) | {overall['substitutions']} / {overall['deletions']} / {overall['insertions']} |
| Reference characters, with / without whitespace | {overall['characters']} / {overall['no_space_characters']} |
| Input exclusions / inference failures | {metrics['input_exclusions']} / {metrics['inference_failures']} |

CER is micro-averaged edit distance/reference characters and may exceed 100%. Exact-tier equality is space-sensitive under the original E1 convention. Leading layout offsets are stripped with lstrip(' '); internal spaces are preserved. The no-space metric removes all whitespace consistently on both sides; it is not Stage 2E's separate collapse-spaces diagnostic. All per-tier predictions/references are in predictions.csv; no samples silently omitted.

## Per source
| Source | Tiers | CER with spaces | CER without whitespace | Exact tiers |
|---|---:|---:|---:|---:|
'''
    for s,m in per_source.items():report+=f"| {s} | {m['tiers']} | {m['cer_including_spaces']:.2%} | {m['cer_excluding_spaces']:.2%} | {m['exact_matches']}/{m['tiers']} |\n"
    report+='\n## Common edit alignments\n'
    for (op,a,b),n in events.most_common(10):report+=f'- {op}: {a!r} -> {b!r}, {n} times\n'
    report+='''
These are minimum-edit alignments using Stage 2E tie preference (substitution, deletion, insertion), not independent semantic diagnoses. Per-role metrics separate upper/lower symbols from count strings in metrics.json.

## Input validation and provenance
The exact Downloads archive handwritten_tier_crops (5).zip contains an internal folder named handwritten_tier_crops (4); the outer version is authoritative, and current files/hashes were checked. Fourteen current validation crop/TXT pairs decode, dimensions match the manifest and all supplied crop hashes match. Labels match manifest source/zero-based row/tier and the existing source JSON. All inputs belong only to all_003191, all_003556 and all_009197. A visual inspection of the 14 current crops shows separated handwritten tiers, not synthetic renders or whole pages. This validates visible tier format, not every Talim symbol's semantic correctness. Faint strokes and residual ruling remain quality limitations.

Only validation crops, their TXT labels, the validation re-photograph, development manifest metadata, report and relevant rendering code were extracted under inputs/. Protected-test image/label members were never opened or extracted. No training images were loaded. Development metadata confirms 88 training-ready and 81 pending training tiers; these statuses were not independently approved by this evaluation.

## all_003191 row 1 correction
The current row-1 upper/count/lower crops have 2026-11-09 correction dates and hashes matching the latest manifest, different from previous hashes. Their source is the supplied real re-photograph, whose SHA-256 begins 7af54bc69e92490f. The current upper crop visibly contains five marks. Older report statements that only two marks are present, auto_flags MARK_GROUPS_2_VS_LABEL_5 and observed_ink_groups=2 are stale metadata; they describe the superseded original-page crop. They do not override the current re-photograph and corrected files. All three row-1 tiers are scored provisionally. Row 0 and row 1 now come from different photographs of the same source; this cannot be presented as one-photo end-to-end OCR.

## Faint-ink rendering limitation
The archived scripts/crop.py render_tier explicitly sets out[(out > 200) & keep] = 255. This can erase actual faint pen ink as well as paper texture. The current all_003556 lower crop uses the documented keep-faint exception; other current crops retain standard rendering lineage. Inspection shows faint/fragmented strokes, especially on all_003556, so quality is a significant confounder of recognition performance. Current pixel statistics are in metrics.json, but counts of surviving bright pixels do not measure erased strokes. The report's earlier 7-validation-crop loss estimate predates the newest row-1 re-photograph and is not claimed as a fresh measurement. Exact current loss requires original tier masks/normalization outputs that are not supplied as reusable masks. No re-segmentation, silent regeneration, threshold change or diagnostic rerender was performed. The primary baseline uses the supplied PNG bytes unchanged.

## Frozen model and validation
'''
    report+=f"E1 checkpoint: {checkpoint}\n\nSHA-256: {release['sha256']} (matches original selected release metadata before/after inference). Vocabulary hash: {state['vocabulary_hash']}. Existing Stage 2E CRNN architecture, strict weights, tokenizer, grayscale /255 normalization, aspect-preserving height-64 resize, collator, true CTC lengths, greedy decoder and edit metrics were reused through context.py and evaluation.evaluate(output=None). No E0/E2 selection. {metrics['validation_tests']['passed']} validation tests passed, {metrics['validation_tests']['failed']} failed.\n"
    report+='''
## Interpretation and readiness
All results remain PROVISIONAL: prior visual QA was performed by an AI assistant, not an independent Talim reader, and file-level agreement does not establish that the writer copied every symbol/count correctly. Crop preparation also used reference row/glyph information, so this is an oracle-assisted manually corrected crop benchmark, never a claim of independent segmentation performance.

The frozen model/input adapter is operational and this validation set can serve as a provisional recognition baseline. The full dataset is not yet ready for an unqualified fine-tuning release: 81 training tiers remain unresolved, faint-ink rendering is inconsistent, internal label spaces follow the synthetic grid rather than handwritten spacing, and semantic verification is incomplete. A separately authorized, controlled pilot using only the 88 currently marked training-ready tiers is technically possible after deciding the space policy, verifying those pairs and rendering train/validation consistently. Do not automatically approve those labels. Preserve this baseline and the protected test split.

Next: independent Talim-reader QA; correct/verify pending training pairs; recover faint ink through a separately versioned, consistent rendering pass; freeze a training-only protocol and compare against this unchanged provisional baseline. No fine-tuning was performed here. Original ZIP, source annotations, E1, Stage 2E/F and earlier Stage 2G results remain unchanged. No Git operations or dashboards.
'''
    (OUT/'baseline_report.md').write_text(report,encoding='utf-8')
    print(json.dumps({'overall':overall,'scored_tiers':len(predictions),'scored_sources':metrics['scored_sources'],'errors':errors,'tests':metrics['validation_tests']},indent=2))
    if not result.wasSuccessful():raise RuntimeError('Validation tests failed; see results')
if __name__=='__main__':main()
