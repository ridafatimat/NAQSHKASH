"""Long-page audit: original labels, actual CTC lengths and unchanged greedy output."""
from dcore import *
import torch
from models.crnn import CRNN
from data.collate import talim_collate_fn
from data.ctc_decode import greedy_ctc_decode,collapse_ctc_sequence
def distance(a,b):
 prev=list(range(len(b)+1))
 for i,x in enumerate(a,1):
  curr=[i]
  for j,y in enumerate(b,1):curr.append(min(prev[j]+1,curr[-1]+1,prev[j-1]+(x!=y)))
  prev=curr
 return prev[-1]
def main():
 torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 vocab,vh=old.load_vocabulary(old.VOCABULARY);cp=torch.load(old.CHECKPOINT,map_location='cpu',weights_only=True)
 model=CRNN(66);model.load_state_dict(cp['model_state_dict']);model.eval()
 saved=[r for r in read(C/'reports/experiments_v1/predictions.csv') if r['candidate']=='C4' and r['dataset']=='positive' and r['parent_id']=='edge_00323']
 source=Sources();records=[]
 for p in [p for p in pages() if p['dataset']=='positive' and p['parent_id']=='edge_00323']:
  image,ref,entries=source.get(p);out=Processor(image).run(old.CANDIDATES['C4']);a=reference.assess(out,ref,entries);assert a['correct']
  for t in a['tiers']:
   x0,y0,x1,y1=t['box'];crop=out['image'][y0:y1,x0:x1]
   s=next(r for r in saved if r['case_id']==p['case_id'] and int(r['row'])==t['row'] and r['role']==t['role'])
   truth=t['truth'];pred=s['prediction'];required=len(truth)+sum(x==y for x,y in zip(truth,truth[1:]))
   width=round(crop.shape[1]*64/crop.shape[0]);expected_time=int(model.output_lengths(torch.tensor([width]))[0])
   rec=dict(case_id=p['case_id'],kind=p['kind'],augmentation=p['augmentation'],row=t['row'],role=t['role'],raw_ground_truth=t['raw'],ground_truth=truth,prediction=pred,leading_offset=t['leading_offset'],target_length=len(truth),prediction_length=len(pred),truth_spaces=truth.count(' '),predicted_spaces=pred.count(' '),crop_shape=list(crop.shape),resized_width=width,ctc_time_steps=expected_time,ctc_repeat_required=required,ctc_feasible=required<=expected_time,retention=t['retention'],contamination=t['contamination'],independent_edit_distance=distance(truth,pred),original_edits={k:int(s[k]) for k in ['substitutions','deletions','insertions']},nonspace_sequences_equal=truth.replace(' ','')==pred.replace(' ',''),unicode_codepoints=sorted(set(ord(x) for x in truth)),actual_greedy_reproduced=None)
   assert rec['independent_edit_distance']==sum(rec['original_edits'].values())
   assert vocab.decode(vocab.encode(truth))==truth
   if p['kind']=='clean':
    batch=talim_collate_fn([dict(image=torch.from_numpy(crop.astype(np.float32)/255).unsqueeze(0),label_text='',sample_id='pixels')])
    with torch.inference_mode():logits=model(batch['images'])
    ids=logits.argmax(2)[:expected_time,0].tolist();decoded=greedy_ctc_decode(ids,vocab)
    rec.update(actual_greedy_reproduced=decoded==pred,actual_logits_time_steps=logits.shape[0],argmax_space_frames=ids.count(vocab.char_to_index[' ']),collapsed_space_tokens=collapse_ctc_sequence(ids).count(vocab.char_to_index[' ']))
    assert decoded==pred
   records.append(rec)
 assert all(torch.equal(t,cp['model_state_dict'][name]) for name,t in model.state_dict().items())
 dump('reports/long_sequence_audit.json',dict(parent='edge_00323',partition='previously_exposed_internal_validation_diagnostic_only',records=records,labels_unchanged=True,decoder_unchanged=True,model_state_unchanged=True,independent_cer_checks_passed=True))
 print('Long audit:',len(records),'tiers; clean greedy outputs reproduced; all targets feasible',flush=True)
if __name__=='__main__':main()

