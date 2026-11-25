"""Read-only local audit. Writes exclusively to a new --out directory."""
import argparse, ast, collections, csv, hashlib, io, json, random, re, zipfile
from pathlib import Path
from PIL import Image, ImageDraw

BASE = Path(r'C:\FYP Dataset Generation')
PACK = BASE/'datasets/talim_real_capture_pilot_source_pack'
PILOT = BASE/'talim_handwritten_real_pilot_10'
ARCHIVE = BASE/'Final dataset/phase2_real_capture_dataset.zip'
REMAIN = BASE/'real_sources/remaining.zip'
def sha(b): return hashlib.sha256(b).hexdigest()
def writecsv(p, rows, fields=None):
    with p.open('w', newline='', encoding='utf-8-sig') as f:
        w=csv.DictWriter(f, fieldnames=fields or list(rows[0])); w.writeheader(); w.writerows(rows)
def sid(s):
    m=re.search(r'all_\d{6}',s); return m.group() if m else None
def inspect_image(b):
    with Image.open(io.BytesIO(b)) as im: im.verify()
    with Image.open(io.BytesIO(b)) as im:
        im.load(); rgb=im.convert('RGB'); return im.size,sha(str(rgb.size).encode()+rgb.tobytes())
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args();out=Path(a.out)
    out.mkdir(parents=True,exist_ok=False)
    (out/'images').mkdir();(out/'annotations').mkdir();(out/'review').mkdir()
    inputs=[]; members=[]; labels=collections.defaultdict(list); source_images=collections.defaultdict(list)
    def record(origin,b,ext):
        inputs.append({'origin':origin,'sha256':sha(b),'bytes':len(b)})
        s=sid(origin)
        if not s:return
        if ext in ('.txt','.json'):labels[s].append((origin,ext,b))
        elif ext in ('.png','.jpg','.jpeg'):
            try: size,pix=inspect_image(b);err=''
            except Exception as e:size=(0,0);pix='';err=str(e)
            source_images[s].append({'origin':origin,'sha256':sha(b),'pixel_sha256':pix,'width':size[0],'height':size[1],'error':err})
    for root in [PACK/'selected_sources',PILOT/'source_ground_truth']:
        for p in sorted(root.rglob('*')):
            if p.is_file():record(str(p),p.read_bytes(),p.suffix.lower())
    with zipfile.ZipFile(REMAIN) as z:
        inputs.append({'origin':str(REMAIN),'sha256':sha(REMAIN.read_bytes()),'bytes':REMAIN.stat().st_size})
        for n in z.namelist():
            if not n.endswith('/'):record(str(REMAIN)+'!'+n,z.read(n),Path(n).suffix.lower())
    captures=list(csv.DictReader((PILOT/'capture_manifest.csv').open(encoding='utf-8-sig')))
    sheets={r['source_id']:r for r in csv.DictReader((PACK/'source_manifest.csv').open(encoding='utf-8-sig'))}
    for p in [PACK/'README.txt',PACK/'source_manifest.csv',PILOT/'README.md',PILOT/'capture_manifest.csv',PILOT/'augmented_manifest_v2.csv',BASE/'Utils/talamNumConverter.py',BASE/'generate_phase3_hard_cases.py',BASE/'generate_individual_distortions.py',BASE/'generate_phase2_realistic_combinations.py',BASE/'generate_remaining_all_3_phases.py']:
        record(str(p),p.read_bytes(),'.metadata')
    mapping=next(ast.literal_eval(n.value) for n in ast.parse((BASE/'Utils/talamNumConverter.py').read_text()).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='num_mapping' for t in n.targets))
    reverse={v:k for k,v in mapping.items() if v}
    with zipfile.ZipFile(ARCHIVE) as z:
        inputs.append({'origin':str(ARCHIVE),'sha256':sha(ARCHIVE.read_bytes()),'bytes':ARCHIVE.stat().st_size})
        ids=sorted({sid(n) for n in z.namelist() if sid(n)})
        # Keep all metadata-associated writer_01 sources together, even though lineage is unverified.
        writer_sources={r['source_id'] for r in captures}
        groups=[sorted(set(ids)&writer_sources)]+[[s] for s in ids if s not in writer_sources]
        groups=[g for g in groups if g];random.Random(20261108).shuffle(groups)
        split={};counts=collections.Counter()
        for g in sorted(groups,key=lambda g:-len(g)):
            dest='train' if len(g)>1 else min(['train','validation','test'],key=lambda k:counts[k]/{'train':24,'validation':3,'test':3}[k])
            for s in g:split[s]=dest
            counts[dest]+=len(g)
        sources=[];issues=[]
        for s in ids:
            variants=labels[s];txt=[x for x in variants if x[1]=='.txt'];js=[x for x in variants if x[1]=='.json'];errors=[];details=[]
            if not txt or not js or not source_images[s]:errors.append('missing_triplet')
            if len({x[2] for x in txt})>1:errors.append('conflicting_txt_copies')
            if len({json.dumps(json.loads(x[2]),sort_keys=True) for x in js})>1:errors.append('conflicting_json_copies')
            for origin,_,b in js:
                try:
                    j=json.loads(b);rows=j['logical_rows']; expected='\n\n'.join('\n'.join(r[k] for k in ['upper','count_line','lower']) for r in rows)
                    if j['sample_id']!=s or j['rows']!=len(rows):errors.append('id_or_row_count')
                    if [r['row_index'] for r in rows]!=list(range(len(rows))):errors.append('tier_order')
                    for _,_,t in txt:
                        # Ignore terminal line delimiters only; preserve all spaces and interior blank lines.
                        if t.decode('utf-8-sig').replace('\r\n','\n').rstrip('\n')!=expected.rstrip('\n'):errors.append('txt_json_exact_mismatch')
                    for r in rows:
                        upper='';lower='';count='';total=0
                        for run in r['runs']:
                            enc=run['encoded_count'];n=sum(reverse[c] for c in enc)
                            if n!=run['count']:errors.append('count_encoding_mismatch')
                            total+=n;count+=enc
                            if run['direction']=='UP':upper+=run['symbol']+' '*(len(enc)-1);lower+=' '*len(enc)
                            elif run['direction']=='DOWN':lower+=run['symbol']+' '*(len(enc)-1);upper+=' '*len(enc)
                            else:errors.append('invalid_direction')
                            if j['symbol_directions'].get(run['symbol'])!=run['direction']:errors.append('symbol_direction_mismatch')
                        if (upper.rstrip(' '),count,lower.rstrip(' '))!=(r['upper'],r['count_line'],r['lower']):errors.append('run_tier_alignment')
                        if total!=j['cells_per_row']:errors.append('cell_count_mismatch')
                    details.append({'origin':origin,'logical_rows':len(rows),'exact_text_sha256':sha(expected.encode())})
                except Exception as e:errors.append('annotation_error:'+str(e))
            for ext,arr in [('.txt',txt),('.json',js)]:
                if arr:(out/'annotations'/f'{s}{ext}').write_bytes(arr[0][2])
            cap=next((r for r in captures if r['source_id']==s),{})
            sources.append({'source_id':s,'split':split[s],'image_copies':len(source_images[s]),'txt_copies':len(txt),'json_copies':len(js),'annotation_valid':not errors,'annotation_issues':';'.join(sorted(set(errors))),'writer_id':'','metadata_capture_writer':cap.get('writer',''),'writer_link_status':'capture metadata exists; archive parent linkage unverified' if cap else 'unknown','original_sheet_id':'','print_template_sheet_id':sheets.get(s,{}).get('print_sheet',''),'provenance':'archive JPG/PNG visually consistent with handwriting; annotation reference PNG synthetic','visual_content_status':'manual_review_required'})
            issues.extend({'source_id':s,'issue':e} for e in sorted(set(errors)))
        bysource={s['source_id']:s for s in sources};seen=collections.defaultdict(list);pixels=collections.defaultdict(list);rows=[];zipnames=collections.Counter(z.namelist())
        for n in z.namelist():
            if n.endswith('/'):continue
            b=z.read(n);name=Path(n).name;s=sid(n);p=out/'images'/name
            if p.exists():raise ValueError('archive basename collision')
            p.write_bytes(b);kind=Path(n).stem.split('__',1)[1] if '__' in name else 'original_photo';err=''
            try:size,pix=inspect_image(b)
            except Exception as e:size=(0,0);pix='';err=str(e)
            seen[sha(b)].append(name)
            if pix:pixels[pix].append(name)
            synthetic=False
            status='excluded' if synthetic or err or not bysource[s]['annotation_valid'] else 'manual_review'
            reason='synthetic clean rendering; outside handwritten-only training' if synthetic else ('unreadable_image:'+err if err else ('annotation_validation_failed' if not bysource[s]['annotation_valid'] else 'full content and exact parent linkage not verified'))
            if any(k in kind for k in ['crop','perspective','rotation','mixed_hard']):reason+='; geometric/crop risk'
            rows.append({'image_path':str(p),'archive_member':n,'source_id':s,'writer_id':'','metadata_capture_writer':bysource[s]['metadata_capture_writer'],'original_sheet_id':'','transcription_path':str(out/'annotations'/f'{s}.txt'),'annotation_path':str(out/'annotations'/f'{s}.json'),'augmentation_type':kind,'split':split[s],'status':status,'reason':reason,'width':size[0],'height':size[1],'sha256':sha(b),'pixel_sha256':pix,'decode_error':err,'provenance':'synthetic_render' if synthetic else 'derived_from_apparent_handwritten_photo; parent_unverified'})
        duplicates={'byte_duplicate_groups':[v for v in seen.values() if len(v)>1],'pixel_duplicate_groups':[v for v in pixels.values() if len(v)>1],'duplicate_archive_members':{k:v for k,v in zipnames.items() if v>1}}
    # Render every image for human review; no OCR or learned models.
    for s in ids:
        rr=[r for r in rows if r['source_id']==s];canvas=Image.new('RGB',(1600,((len(rr)+3)//4)*290),'#ddd');d=ImageDraw.Draw(canvas)
        for i,r in enumerate(rr):
            x=i%4*400;y=i//4*290;d.text((x+4,y+4),r['augmentation_type'],fill='black')
            if not r['decode_error']:
                with Image.open(r['image_path']) as im:im.thumbnail((390,255));canvas.paste(im.convert('RGB'),(x,y+28))
        canvas.save(out/'review'/f'{s}.jpg',quality=90)
    writecsv(out/'manifest.csv',rows);writecsv(out/'accepted_manifest.csv',[r for r in rows if r['status']=='accepted'],list(rows[0]));writecsv(out/'source_split.csv',sources)
    writecsv(out/'review_queue.csv',[dict(r,review_decision='',reviewer='',review_notes='') for r in rows if r['status']=='manual_review'])
    writecsv(out/'input_hashes.csv',inputs)
    (out/'source_image_inventory.json').write_text(json.dumps(source_images,indent=2),encoding='utf-8')
    (out/'duplicates.json').write_text(json.dumps(duplicates,indent=2),encoding='utf-8')
    (out/'annotation_issues.json').write_text(json.dumps(issues,indent=2),encoding='utf-8')
    summary={'images':len(rows),'sources':len(ids),'extensions':dict(collections.Counter(Path(r['image_path']).suffix for r in rows)),'status':dict(collections.Counter(r['status'] for r in rows)),'source_splits':dict(counts),'annotation_valid_sources':sum(s['annotation_valid'] for s in sources),'unreadable_images':sum(bool(r['decode_error']) for r in rows),'source_image_errors':sum(bool(x['error']) for v in source_images.values() for x in v),'duplicate_byte_groups':len(duplicates['byte_duplicate_groups']),'duplicate_pixel_groups':len(duplicates['pixel_duplicate_groups'])}
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (out/'audit_report.md').write_text('# Stage 2G dataset preparation\n\n'+json.dumps(summary,indent=2)+'\n\nAll originals were read only. Only this isolated output directory was written. No Git, OCR, model imports, training, fine-tuning or protected-test evaluation. E1 and Stage 2F were not opened or changed.\n\n## Annotation validation\nAll annotation copies in all three collections were compared. UTF-8 BOM and CRLF are transport-normalized and final newline delimiters ignored; no spaces are stripped from TXT. Tier order, internal blank lines, symbols, run alignment, direction, encoded numeric counts and row cell totals are checked. See annotation_issues.json; duplicated annotation copies are not extra sources. Source image copies and hashes are in source_image_inventory.json. The remaining archive contains 10 IDs, six overlapping the 20-ID pack; source_ground_truth contains 20 IDs, 14 overlapping that union, adding six.\n\n## Provenance and identity\nAll 30 brightness representatives were visually inspected together: irregular pen strokes on notebook/plain paper are consistent with handwriting. No photographed printed example was positively identified; this is visual evidence, not verified chain of custody. Source-pack README and pilot README explicitly identify clean source PNGs as synthetic; JSON rendering metadata names Talamat.ttf/IMTCRPET. The archive PNGs were separately viewed and are handwritten photographs, not these synthetic annotation references; they require manual review too. Writer_01 is recorded for 10 raw captures only. Archive lacks parent capture mapping, so writer_id stays blank and metadata_capture_writer preserves the limited claim separately. No original physical sheet identity is documented; print-template sheet IDs must not be interpreted as handwritten sheet identities. Full provenance confirmation remains manual.\n\n## Content and acceptance\nPillow verify plus full decode is run on every dataset and source image; hashes identify exact byte and decoded-pixel duplicates. No perceptual duplicate or semantic completeness claim is made. Every JPG requires side-by-side full-resolution inspection against its original capture and TXT/JSON, including counts and all tiers. Contact sheets cover all 920 images. Partial crops, fixed-canvas rotation, perspective and mixed-hard variants receive explicit risk flags. Available generation scripts are hashed as contextual evidence only: archive lacks recipes/seeds/transform matrices linking each image, so script names cannot prove content preservation. Partial crops cannot safely inherit full-image labels without checking removed regions. Existing photos can already contain clipped edge marks. No JPG is automatically accepted. accepted_manifest.csv is deliberately header-only until manual review; manifest.csv is a complete preparation manifest, not an approved training release. Do not train directly from unfiltered manifest.csv.\n\n## Split\nDeterministic seed 20261108. Target 24/3/3 sources. All variants and original photographs for each source remain in one split. Ten sources associated with writer_01 capture metadata are additionally grouped together conservatively. Writer linkage for other sources is unknown: this is NOT writer-independent or sheet-independent evaluation. No protected test set was accessed. These are newly proposed dataset partitions only. Duplicate groups must be resolved before release, especially any cross-source duplicates.\n\n## Review\nUse source_split.csv, review_queue.csv, review/*.jpg, original inputs and full-resolution images. Record reviewer, decision and notes in a COPY of review_queue.csv. Require verified full Talim content, label agreement, provenance and parent linkage before acceptance. Exclude content-losing crops with full labels or create separately reviewed partial labels in a later stage. This preparation does not automatically consume review decisions.\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
