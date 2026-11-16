"""Complete and freeze diagnosis before targeted implementation."""
from dcore import *
from collections import Counter,defaultdict
import html
def mdtable(headers,rows):
 return '| '+' | '.join(headers)+' |\n|'+'|'.join(['---']*len(headers))+'|\n'+'\n'.join('| '+' | '.join(map(str,r))+' |' for r in rows)
def main():
 compare=read(ROOT/'reports/c4_c5_comparison.csv');traces=json.loads((ROOT/'reports/role_traces.json').read_text())
 replay=json.loads((ROOT/'reports/reproduction.json').read_text());long=json.loads((ROOT/'reports/long_sequence_audit.json').read_text())
 counts=[]
 for part in ['tuning','validation','clean_regression']:
  for ds in ['positive','general','normal']:
   for kind in ['scanned','clean']:
    group=[r for r in compare if r['partition']==part and r['dataset']==ds and r['kind']==kind]
    if not group:continue
    for pair in sorted({(r['C4'],r['C5']) for r in group}):
     rs=[r for r in group if (r['C4'],r['C5'])==pair]
     counts.append(dict(partition=part,dataset=ds,kind=kind,C4=pair[0],C5=pair[1],images=len(rs),parents=len({r['parent_id'] for r in rs})))
 csvout('reports/comparison_counts.csv',counts)
 tuning=[t for t in traces if t['partition']=='tuning'];signatures=[]
 for name,fn in [
  ('wrong proposal despite sole legal path (margin sentinel 99)',lambda t:t['accepted'] and not t['correct'] and t['diagnostics']['role_margin']==99),
  ('scale smaller than 65% of tallest band',lambda t:t['diagnostics']['component_scale']<.65*max(b-a for a,b in t['diagnostics']['bands'])),
  ('band taller than 1.6 component scales',lambda t:any(b-a>1.6*t['diagnostics']['component_scale'] for a,b in t['diagnostics']['bands'])),
  ('wrong row count',lambda t:t['accepted'] and len(t['proposed_rows'])!=len(t['expected_roles'])),
  ('accepted wrong structure',lambda t:t['accepted'] and not t['correct'])]:
  for candidate in ['C4','C5']:
   g=[t for t in tuning if t['candidate']==candidate and fn(t)]
   signatures.append(dict(candidate=candidate,signature=name,images=len(g),parents=len({(t['dataset'],t['parent_id']) for t in g}),representative=g[0]['case_id'] if g else ''))
 csvout('reports/diagnostic_signatures.csv',signatures)
 chosen=[];seen=set()
 for event in [('wrong','wrong'),('wrong','rejected'),('correct','rejected'),('correct','correct'),('wrong','correct')]:
  for ds in ['positive','general']:
   rs=[r for r in compare if r['partition']=='tuning' and r['dataset']==ds and (r['C4'],r['C5'])==event]
   for r in rs[:2]:
    if key(r) not in seen:chosen.append(r);seen.add(key(r))
 chosen+= [p for p in pages() if p['parent_id']=='edge_00323' and p['kind']=='clean']
 src=Sources();allpages={key(p):p for p in pages()};gallery=[]
 for r in chosen:
  p=allpages[key(r)];image,ref,entries=src.get(p);processor=Processor(image)
  folder=ROOT/'galleries'/(p['dataset']+'_'+p['case_id']);folder.mkdir(exist_ok=True)
  cv2.imwrite(str(folder/'original.png'),image)
  info=dict(page=p,expected=[{k:e[k] for k in ['row','role','truth']} for e in entries],candidates={})
  for name in ['C4','C5']:
   out=processor.run(old.CANDIDATES[name]);a=reference.assess(out,ref,entries)
   panel=cv2.cvtColor(out['image'],cv2.COLOR_GRAY2BGR)
   for row in out['rows']:
    for t in row:
     x0,y0,x1,y1=t['box'];color=(0,140,0) if a['correct'] else (0,0,220)
     cv2.rectangle(panel,(x0,y0),(x1-1,y1-1),color,1);cv2.putText(panel,t['role'],(x0,max(10,y0-2)),cv2.FONT_HERSHEY_SIMPLEX,.35,color,1)
   cv2.imwrite(str(folder/(name+'.png')),panel)
   cv2.imwrite(str(folder/(name+'_detail.png')),panel[:,:min(1000,panel.shape[1])])
   info['candidates'][name]=dict(accepted=out['accepted'],assessment=a,rows=out['rows'],diagnostics=out['diagnostics'])
  old.write_json(folder/'evidence.json',info);gallery.append(dict(folder=folder.name,case=p['case_id'],partition=p['partition']))
 dump('galleries/selection.json',gallery)
 rows_html=[]
 for g in gallery:
  name=html.escape(g['case']);folder=g['folder']
  imgs=''.join(f'<figure><figcaption>{c}</figcaption><a href="{folder}/{c}.png"><img src="{folder}/{c}.png"></a></figure>' for c in ['original','C4','C5'])
  rows_html.append(f'<section><h2>{name}</h2><p>{g["partition"]}; <a href="{folder}/evidence.json">ground truth, assignments, diagnostics</a></p><div>{imgs}</div></section>')
 (ROOT/'galleries/index.html').write_text('<!doctype html><meta charset="utf-8"><title>Stage 2D segmentation diagnosis</title><style>body{font:16px sans-serif;margin:24px}section{border-top:1px solid #aaa}section div{display:flex;gap:12px}figure{margin:0;max-width:32%}img{width:100%;max-height:650px;object-fit:contain;object-position:top}a{color:#145}pre{white-space:pre-wrap}</style><h1>Stage 2D diagnostic gallery</h1><p>Full resolution: click images. No labels enter segmentation. Rejected outputs have no accepted boxes; reference and failure reasons are in evidence sidecars.</p>'+''.join(rows_html),encoding='utf-8')
 table=mdtable(['Partition','Dataset','Kind','C4','C5','Images','Unique parents'],[[r[k] for k in ['partition','dataset','kind','C4','C5','images','parents']] for r in counts])
 sigtable=mdtable(['Candidate','Diagnostic signature (not automatically a root cause)','Images','Parents','Example'],[[r[k] for k in ['candidate','signature','images','parents','representative']] for r in signatures])
 clean=[r for r in long['records'] if r['kind']=='clean']
 report=f"""# Stage 2D failure diagnosis

Diagnosis completed before D1 implementation. Stage 2C remains unchanged.

## Evidence and counting
All 486 inventoried artifacts and source/software locks passed. Recomputed cohort summaries from 9,672 saved page outcomes and 31,367 predictions. Acceptance and reference correctness were reproduced for {replay['candidate_outcomes']} C4/C5 outcomes on {replay['pages']} pages. Character errors were independently checked for the long parent. General sources remain exploratory; all old validation scores are exposed development comparisons, never a new independent test.

{table}

Counts are images and unique source parents within each cell. Parent counts across cells overlap; do not add them as independent observations. Every page is a valid positive Talim source; every rejection is a missed valid page, not a negative example.

## Ranked issues supported by tuning data
1. **Wrong scale and disconnected glyph grouping.** Component-height percentile can be 8–23 pixels while the true count-band extent is 38–42 pixels. In edge_00025, the 41-pixel count and 23-pixel lower tier are forced into separate rows because the inferred pitch is too short. In edge_00001, separated dot clusters of the upper A remain separate bands. Correct paths become infeasible before confidence is assessed. Fix the scale estimate using observed band extent and constrain merges by total extent. This has high recovery potential, but must be tested on sparse unusual layouts.
2. **Indiscriminate small-gap merges.** General rotation/perspective tuning examples (all_000655, all_004700) join a full count tier to its lower symbol tier into an 80+-pixel band at scale ~38. A 6-pixel gap rule alone cannot distinguish fragments from full tiers. Add an extent guard and reject unresolved over-tall bands; do not blindly split ink.
3. **Heuristic role evidence and misleading certainty.** A single legal path receives margin=99 even when morphology is poor or the true path was pruned by geometry. This sentinel is not calibrated confidence. C5 only tests the best/second-path margin and therefore still accepts confidently wrong layouts. Expose alternative count, score and structural checks; retain ambiguity rejection without describing scores as probabilities.
4. **Role morphology for disconnected count glyphs.** Count glyphs may consist of individually short components; component-height alone underestimates the whole symbol. Evaluate a bounded band-height contribution using tuning examples, keeping all original pixels.
5. **Long recognition and spacing.** edge_00323 is confirmed in the saved artifacts. Its clean crops reproduce the saved greedy predictions with {sum(r['independent_edit_distance'] for r in clean)} errors, all missing internal spaces; non-space sequences match. Its ground-truth labels round-trip through the unchanged vocabulary; independently computed edit distances agree, repeat-aware CTC lengths are feasible, and reference crop retention passes. The space tokens are already missing in the collapsed network argmax output. This supports a frozen recognizer/spacing limitation, not a CER normalization or encoding bug. Degraded variants also have genuine glyph errors. No space-stripping or arbitrary crop workaround is justified.

## Role search audit
The exact Stage 2C scoring uses optional upper, mandatory count, optional lower; row penalty -1.4; log-transformed logistic morphology evidence; geometric pitch and boundary constraints. C4 accepts the best legal path regardless of ambiguity; C5 rejects margin <0.2. Two paths per index suffice because future contributions depend only on the band index, not past path state.
Exhaustive enumeration matched the top two scores on 1,350 seeded sequences up to nine bands and small actual replay sequences. **No verified DP search defect.** Scoring and invalid geometric constraints can still produce the wrong highest-scoring path; that is a separate issue.

{sigtable}

These signatures overlap and are not causal labels. failure_categories.csv retains actual first failure reasons. Geometry/background changes cannot be blamed merely because a transformed image failed. Stage 2C's registered component audit found no component loss among its aligned cases, but it did not cover wrong/rejected structures. D1 will audit component retention separately; no universal preservation claim follows.

## Long-page audit and limits
Exact truth/prediction strings, raw leading offsets, dimensions, Unicode values, target lengths, required repeats, network output lengths, and clean argmax/decoded space counts are in long_sequence_audit.json. Row/tier order comes from the frozen annotation reference. Meaningful internal spaces remain in all headline CER calculations. The diagnostic strips spaces only for equality attribution. Offline ideal-ink masks approximate blur/interpolation and are not a human verification of every pixel.

## Targeted experiment decision
Build on C5's normalization/geometry/denoising, retain C4/C5 unchanged. Change only the grouping/role engine: robust band-aware scale, extent-limited fragment merging, bounded whole-band morphology, explicit confidence diagnostics and structured abstention. Compare a small predeclared grid on original tuning parents only; freeze the selected D1 before replaying exposed validation. No untouched positive holdout is used or claimed. Long OCR behavior is documented for subsequent training design; no recognition formatting candidate is justified.

Gallery: [index](../galleries/index.html). Source data, labels, model, original decoder and historical files are read-only.
"""
 (ROOT/'reports/failure_analysis.md').write_text(report,encoding='utf-8')
 dump('reports/diagnosis_complete.json',dict(complete=True,before_D1_implementation=not (ROOT/'engine.py').exists(),report_sha256=old.digest(ROOT/'reports/failure_analysis.md'),search_audit= json.loads((ROOT/'reports/search_audit.json').read_text()),reproduction=replay))
 print('Diagnosis completed; source-parent counts and galleries saved',flush=True)
if __name__=='__main__':main()

