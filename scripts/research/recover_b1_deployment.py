"""Resume-capable fine tuning / CPU-targeted QAT using TRAIN only.

Selection uses validation of the actual deployment model. Test is never opened.
QAT exports a quantized TorchScript CPU model, not a TensorRT GPU engine.
"""
import argparse
import copy
import json
from pathlib import Path
import random
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from src.straw_global_local.data import file_hash
from src.straw_global_local.deployment import TensorB1,packet,decode
from src.straw_global_local.runner import classification_metrics,atomic_torch_save

ROOT=Path(__file__).resolve().parents[2]
CHECKPOINT=ROOT/'artifacts/local_b1_check/B1_s42/best.pt'


def qat_model(checkpoint):
    from torchvision.models.quantization import resnet18
    from torch.ao.quantization import get_default_qat_qconfig,prepare_qat
    supported=torch.backends.quantized.supported_engines
    engine=next((e for e in ('x86','fbgemm','onednn') if e in supported),None)
    if engine is None: raise RuntimeError('No supported CPU INT8 engine in this PyTorch build')
    torch.backends.quantized.engine=engine
    base=TensorB1(checkpoint)
    model=resnet18(weights=None,quantize=False,num_classes=5)
    state=dict(base.encoder.state_dict())
    state['fc.weight']=base.classifier.weight.detach()
    state['fc.bias']=base.classifier.bias.detach()
    model.load_state_dict(state,strict=True)
    model.train()
    model.fuse_model(is_qat=True)
    model.qconfig=get_default_qat_qconfig(engine)
    prepare_qat(model,inplace=True)
    return model


def converted(model):
    from torch.ao.quantization import convert
    return convert(copy.deepcopy(model).cpu().eval(),inplace=True)


class Inputs(Dataset):
    def __init__(self,frame,size,quality,augment):
        self.frame=frame.reset_index(drop=True); self.size=size; self.quality=quality; self.augment=augment
        self.cache=ROOT/f'artifacts/b1_deployment/input_cache/{size}_q{quality}'
        self.cache.mkdir(parents=True,exist_ok=True)
    def __len__(self): return len(self.frame)
    def __getitem__(self,i):
        row=self.frame.iloc[i]; path=self.cache/f'{row.sample_id}.jpg'
        if not path.exists():
            original=ROOT/'data/straw_defects_raw'/row.relative_path
            if file_hash(original)!=row.sha256: raise ValueError('Input hash mismatch')
            tmp=path.with_suffix('.tmp'); tmp.write_bytes(packet(original,self.size,self.quality)); tmp.replace(path)
        x=torch.from_numpy(decode(path.read_bytes(),self.size)[0])
        if self.augment:
            if torch.rand(())<.5: x=x.flip(-1)
            mean=torch.tensor([.485,.456,.406])[:,None,None]
            std=torch.tensor([.229,.224,.225])[:,None,None]
            x=(((x*std+mean)*(.9+.2*torch.rand(()))).clamp(0,1)-mean)/std
        return x,int(row.label)


@torch.inference_mode()
def validate(model,loader,device):
    model.eval(); scores=[]; labels=[]
    for x,y in loader:
        scores.append(model(x.to(device)).float().cpu().numpy()); labels.extend(y.numpy())
    logits=np.concatenate(scores)
    return classification_metrics(np.asarray(labels),logits),logits,np.asarray(labels)


def main():
    global CHECKPOINT
    parser=argparse.ArgumentParser()
    parser.add_argument('--method',choices=['finetune','qat'],required=True)
    parser.add_argument('--size',type=int,choices=[384,512],default=512)
    parser.add_argument('--quality',type=int,default=90)
    parser.add_argument('--epochs',type=int,default=5)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--tag',default='',help='Separate reproducible recovery run; letters/digits/hyphens only')
    parser.add_argument('--smoke',action='store_true',help='One update / tiny validation; not a quality result')
    args=parser.parse_args()
    if args.tag and (not args.tag.replace('-','').isalnum()):raise ValueError('Unsafe run tag')
    CHECKPOINT=ROOT/f'artifacts/local_b1_check/B1_s{args.seed}/best.pt'
    torch.set_num_threads(4); torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    train_path=ROOT/'artifacts/straw_global_local/data_v1/train.csv'
    val_path=ROOT/'artifacts/straw_global_local/data_v1/val.csv'
    frames={r:pd.read_csv(p) for r,p in [('train',train_path),('val',val_path)]}
    options={k:v for k,v in vars(args).items() if k!='tag' or v}
    spec={**options,'checkpoint_sha256':file_hash(CHECKPOINT),'train_sha256':file_hash(train_path),'val_sha256':file_hash(val_path)}
    folder=ROOT/f'artifacts/b1_deployment/recovery_{args.method}_{args.size}_s{args.seed}{"_smoke" if args.smoke else ""}{"_"+args.tag if args.tag else ""}'
    folder.mkdir(parents=True,exist_ok=True)
    record=folder/'spec.json'
    if record.exists() and json.loads(record.read_text())!=spec: raise ValueError('Recovery configuration changed; use another folder')
    record.write_text(json.dumps(spec,indent=2),encoding='utf-8')
    if args.smoke:
        frames={r:f.groupby('label').head(1) for r,f in frames.items()}
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=qat_model(CHECKPOINT) if args.method=='qat' else TensorB1(CHECKPOINT)
    model=model.to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=1e-4)
    generator=torch.Generator().manual_seed(args.seed+17)
    train=DataLoader(Inputs(frames['train'],args.size,args.quality,True),batch_size=2,shuffle=True,generator=generator)
    val=DataLoader(Inputs(frames['val'],args.size,args.quality,False),batch_size=4)
    start,best,history=0,-1.,[]
    eligible_best=-1.
    reference=json.loads((ROOT/'artifacts/b1_deployment/torch_fp32_exact/result.json').read_text())
    last=folder/'last.pt'
    if last.exists():
        state=torch.load(last,map_location='cpu',weights_only=False)
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        start,best,history=state['epoch']+1,state['best'],state['history']
        eligible_best=state.get('eligible_best',-1.)
        torch.set_rng_state(state['rng']); generator.set_state(state['loader_rng'])
        random.setstate(state['python_rng']); np.random.set_state(state['numpy_rng'])
        if device=='cuda': torch.cuda.set_rng_state_all(state['cuda_rng'])
    for epoch in range(start,1 if args.smoke else args.epochs):
        model.train()
        # Preserve frozen BN behavior of the source B1; QAT observers stay enabled initially.
        for layer in model.modules():
            if isinstance(layer,nn.modules.batchnorm._BatchNorm): layer.eval()
        if args.method=='qat':
            model.apply(torch.ao.nn.intrinsic.qat.freeze_bn_stats)
            if epoch>=max(1,args.epochs-2): model.apply(torch.ao.quantization.disable_observer)
        optimizer.zero_grad(set_to_none=True)
        for i,(x,y) in enumerate(train):
            # Average each four-batch window, including the shorter final window.
            window=i//4*4; denominator=min(8,len(train.dataset)-window*train.batch_size)
            loss=nn.functional.cross_entropy(model(x.to(device)),y.to(device),reduction='sum')/denominator
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
            loss.backward()
            if (i+1)%4==0 or i+1==len(train) or args.smoke: optimizer.step();optimizer.zero_grad(set_to_none=True)
            if (i+1)%100==0: print(args.method,'epoch',epoch,'batch',i+1,'/',len(train),flush=True)
            if args.smoke: break
        deployment=converted(model) if args.method=='qat' else copy.deepcopy(model).eval()
        result,logits,labels=validate(deployment,val,'cpu' if args.method=='qat' else device)
        eligible=(not args.smoke and result['macro_f1']>=reference['macro_f1']-.005 and result['fn']<=reference['fn'] and result['fp']<=reference['fp'])
        history.append(dict(epoch=epoch,validation=result,eligible=eligible))
        if eligible and result['macro_f1']>eligible_best:
            eligible_best=result['macro_f1']
            if args.method=='qat':torch.jit.save(torch.jit.script(deployment),str(folder/'best_eligible_int8.ts'))
            else:atomic_torch_save(folder/'best_eligible.pt',{'model':model.state_dict(),'spec':spec,'validation':result})
            np.savez_compressed(folder/'eligible_val_predictions.npz',logits=logits,labels=labels)
        if result['macro_f1']>best:
            best=result['macro_f1']
            if args.method=='qat':
                torch.jit.save(torch.jit.script(deployment),str(folder/'best_int8.ts'))
            else:
                atomic_torch_save(folder/'best.pt',{'model':model.state_dict(),'spec':spec,'validation':result})
            np.savez_compressed(folder/'best_val_predictions.npz',logits=logits,labels=labels)
        atomic_torch_save(last,dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,best=best,eligible_best=eligible_best,history=history,
            rng=torch.get_rng_state(),loader_rng=generator.get_state(),python_rng=random.getstate(),numpy_rng=np.random.get_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if device=='cuda' else None))
        (folder/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
        print(args.method,'epoch',epoch,'val_f1',result['macro_f1'],'FN',result['fn'],'FP',result['fp'],flush=True)
    print('Recovery complete: '+str(folder),flush=True)


if __name__=='__main__': main()
