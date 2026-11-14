# Construction and recognition source evidence

Sources were read, not executed. The original generator is retained in the project staging workspace; the deployed dataset_tools folder lacks that script.

## C:\Users\hp\OneDrive\Documents\ChatGPT\FYP\edge_audit_staging\dataset_tools\edge_case_audit.py

SHA-256: 0bd527ea2359d9bac8ae20493d84cc02c89f784e208f0889e9009a6e1647397a

```text
227: def negatives(folder):
228:     base=make([[(UP[0],11),(DOWN[0],19)],[(DOWN[1],20),(UP[1],10)]],'negative')
229:     manifest=[]
230:     def case(name,expected,mutate):
231:         path=folder/name; path.mkdir(parents=True)
232:         s=copy.deepcopy(base); write_sample(path,s); write_csv(path,[s])
233:         mutate(path,s)
234:         try:
235:             result=validate_dataset_folder(path); errors=result['failures']
236:             ok=result['status']=='FAIL' and any(expected in e for e in errors)
237:         except Exception as exc:
238:             errors=['CRASH '+repr(exc)]; ok=False
239:         manifest.append(dict(sample_id=name,violated_rule=name,mutation_applied=name,source_valid_sample='embedded base: A11/R19, S20/B10',expected_error=expected,actual_errors=errors,passed=ok))
240:     def mutate_json(fn):
241:         def apply(path,s):
242:             fn(s); dump(path/'json/negative.json',s)
243:         return apply
244:     def runfield(field,value): return mutate_json(lambda s:s['logical_rows'][0]['runs'][0].__setitem__(field,value))
245:     for name,value in [('zero',0),('negative',-1),('above_max',max(derive(ROOT)[0])+1),('bool',True),('float',1.2),('string','11')]:
246:         case('count_'+name,'unsupported count',runfield('count',value))
247:     case('wrong_encoding','invalid encoded_count',runfield('encoded_count','a'))
248:     case('unsupported_token','invalid encoded_count',runfield('encoded_count','?'))
249:     for name,value in [('unsupported','?'),('ambiguous','O'),('type',[])]:
250:         case('symbol_'+name,'expected string' if name=='type' else 'unsupported symbol',runfield('symbol',value))
251:     case('wrong_direction','direction violates',runfield('direction','DOWN'))
252:     case('invalid_direction','invalid direction',runfield('direction','LEFT'))
253:     case('adjacent_identical','adjacent identical',mutate_json(lambda s:s['logical_rows'][0]['runs'][1].__setitem__('symbol',UP[0])))
254:     for delta in (-1,1): case('row_sum_'+str(delta),'run counts sum',mutate_json(lambda s,d=delta:s.__setitem__('cells_per_row',30+d)))
255:     case('empty_row','empty or malformed runs',mutate_json(lambda s:s['logical_rows'][0].__setitem__('runs',[])))
256:     case('empty_sample','no logical_rows',mutate_json(lambda s:s.__setitem__('logical_rows',[])))
257:     for name,fn in [('wrong_line',lambda r:r.__setitem__('lower',r['upper'])),('missing_symbol',lambda r:r.__setitem__('upper','')),('missing_count',lambda r:r.__setitem__('count_line','')),('repeated_symbol_alignment',lambda r:r.__setitem__('upper','AA')),('trailing_whitespace',lambda r:r.__setitem__('upper',r['upper']+' '))]:
258:         case(name,'canonical upper/count/lower mismatch',mutate_json(lambda s,f=fn:f(s['logical_rows'][0])))
259:     for name,text in [('txt_mismatch','wrong'),('row_separator',canonical_text(base).replace('\n\n','\n',1)),('extra_line',canonical_text(base)+'x\n')]:
260:         case(name,'TXT/JSON',lambda p,s,t=text:(p/'labels/negative.txt').write_text(t,encoding='utf-8'))
261:     case('actual_symbols','actual_symbols_used mismatch',mutate_json(lambda s:s.__setitem__('actual_symbols_used',[])))
262:     case('symbol_pool','symbol_pool mismatch',mutate_json(lambda s:s.__setitem__('symbol_pool',[])))
263:     for field in ('sample_id','rows','cells_per_row','symbols_used','symbol_directions','logical_rows','generation','rendering'):
264:         expected={'sample_id':'invalid sample_id','rows':'rows field','cells_per_row':'cells_per_row','symbols_used':'symbols_used','symbol_directions':'symbol_directions','logical_rows':'no logical_rows','generation':'generation','rendering':'rendering'}[field]
265:         case('missing_'+field,expected,mutate_json(lambda s,k=field:s.pop(k)))
266:     for name,fn,expected in [('runs_type',lambda s:s['logical_rows'][0].__setitem__('runs',5),'malformed runs'),('row_type',lambda s:s['logical_rows'].__setitem__(0,7),'expected object')]:
267:         case(name,expected,mutate_json(fn))
268:     for name,sub,suffix,expected in [('png','images','png','PNG missing'),('txt','labels','txt','TXT label missing'),('json','json','json','no JSON')]:
269:         case('missing_'+name+'_artifact',expected,lambda p,s,d=sub,x=suffix:(p/d/('negative.'+x)).unlink())
270:     case('corrupt_png','PNG unreadable',lambda p,s:(p/'images/negative.png').write_bytes(b'not a PNG'))
271:     case('blank_png','PNG has no ink',lambda p,s:Image.new('L',(s['rendering']['image_width_px'],s['rendering']['image_height_px']),255).save(p/'images/negative.png'))
272:     case('wrong_image_mode','PNG mode',lambda p,s:Image.open(p/'images/negative.png').convert('RGB').save(p/'images/negative.png'))
273:     case('wrong_dimensions','PNG dimensions',lambda p,s:Image.new('L',(2,2),0).save(p/'images/negative.png'))
274:     for name,raw in [('malformed_json','{'),('broken_escape','{"symbol":"\\q"}')]:
275:         case(name,'malformed JSON',lambda p,s,t=raw:(p/'json/negative.json').write_text(t,encoding='utf-8'))
276:     def duplicate(p,s):
277:         second=copy.deepcopy(s); second['sample_id']='second'; write_sample(p,second); write_csv(p,[s,second])
278:     case('semantic_duplicate','semantic duplicate',duplicate)
```

## C:\NAQSHKASH\stage2\stage2f_negative_rejection\src\evaluate_baseline.py

SHA-256: b1ff697a09be9cd0487f8808fa62a541d8901644a238eeddb3d7dc46f3ce56c7

```text
90:         return None
91:     source = record['json_path']
92:     if record['status'] == 'negative_fixture' and not ('duplicate_id' in source and 'second' in source):
93:         source = str(DATA / 'talim_edge_cases_negative_v1/actual_symbols/json/negative.json')
94:     try:
95:         sample = json.loads(text_asset(source))
96:         from talim.talim_rules import canonical_row
97:         return [{role: text.lstrip(' ') for role, text in zip(('upper', 'count', 'lower'), canonical_row(r['runs'])) if text.strip()}
98:                 for r in sample['logical_rows']]
99:     except (OSError, KeyError, TypeError, ValueError):
100:         return None
101: 
102: 
103: def main():
```

## C:\NAQSHKASH\stage2\stage2d_experiments\engine.py

SHA-256: c29e6f09c7a15c72041a6c68d9827c2b04885e37eab1e60c1d14a1fb3214996d

```text
90: 
```
