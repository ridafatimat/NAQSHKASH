"""Real small optimization sanity check; never a selected trained model."""
from context import *
from datasets import FrozenTiers
from checkpoint_utils import save_checkpoint,set_finetuning_mode,forward_training
def main():
 seed();verify();m,v,state=model_load();optimizer=torch.optim.AdamW(m.parameters(),lr=1e-5,weight_decay=1e-4);encoder=CTCLabelEncoder(v);criterion=torch.nn.CTCLoss(blank=0,zero_infinity=False)
 positive=read(E/'manifests/positive_train.csv');replay=read(E/'manifests/synthetic_replay_train.csv')
 chosen=[min(positive,key=lambda r:int(r['resized_width'])),replay[0],max(positive,key=lambda r:int(r['resized_width']))]
 records=[]
 for r in chosen:
  before=time.perf_counter();dataset=FrozenTiers([r]);batch=talim_collate_fn([dataset[0]]);enc=encoder.encode_batch(batch['label_texts']);set_finetuning_mode(m)
  optimizer.zero_grad(set_to_none=True);oldweight=m.classifier.weight.detach().clone()
  logits=forward_training(m,batch['images']);lengths,required=runtime_ctc_lengths(m,batch,enc,logits.shape[0])
  loss,_=fp32_ctc_loss(logits,enc['targets'],lengths,enc['target_lengths'],criterion);assert torch.isfinite(loss);loss.backward()
  norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.);assert torch.isfinite(norm);optimizer.step();assert not torch.equal(oldweight,m.classifier.weight)
  pred=greedy_ctc_decode(logits.argmax(2)[:int(lengths[0]),0].tolist(),v);assert set(pred)<=set(v.char_to_index)
  records.append(dict(id=r['id'],cohort=r['cohort'],resized_width=r['resized_width'],loss=float(loss.detach()),gradient_norm=float(norm),weight_updated=True,seconds=time.perf_counter()-before))
  print('PREFLIGHT',json.dumps(records[-1]),flush=True)
 path=E/'checkpoints/preflight/not_selected.pt';h=save_checkpoint(path,m,optimizer,dict(role='sanity_check_not_selected'),3,v)
 loaded,_,saved=model_load(path);m.eval();loaded.eval()
 small=talim_collate_fn([FrozenTiers([chosen[0]])[0]])['images']
 with torch.inference_mode():assert torch.equal(m(small),loaded(small))
 assert digest(CHECKPOINT)==EXPECTED
 dump('reports/preflight.json',dict(records=records,reloaded_logits_equal=True,checkpoint=str(path),sha256=h,original_unchanged=True,selected=False,passed=True))
 print('PREFLIGHT COMPLETE',flush=True)
if __name__=='__main__':main()

