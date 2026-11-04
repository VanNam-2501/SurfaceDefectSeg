"""B1 deployment building blocks. No random-weight fallback and no test tuning."""
from __future__ import annotations
import io
import time
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from .model import GlobalLocalCNN

MEAN = np.asarray([.485,.456,.406],np.float32)[None,:,None,None]
STD = np.asarray([.229,.224,.225],np.float32)[None,:,None,None]


class TensorB1(nn.Module):
    def __init__(self, checkpoint):
        super().__init__()
        saved = torch.load(Path(checkpoint),map_location='cpu',weights_only=False)
        base = GlobalLocalCNN('B1',pretrained=False).eval()
        base.load_state_dict(saved['model'],strict=True)
        self.encoder, self.classifier = base.encoder, base.classifier

    def forward(self,x):
        return self.classifier(self.encoder(x))


def image_array(image, size=512):
    image = ImageOps.exif_transpose(image).convert('RGB')
    if image.size != (size,size):
        image = image.resize((size,size),Image.Resampling.BILINEAR)
    return np.asarray(image).copy()


def normalize(rgb):
    x = np.ascontiguousarray(rgb.transpose(2,0,1)[None],dtype=np.float32)
    x /= 255.
    x -= MEAN
    x /= STD
    return x


def decode(payload,size=512):
    with Image.open(io.BytesIO(payload)) as image:
        return normalize(image_array(image,size))


def packet(path,size=512,quality=90):
    """Offline proxy for browser resize/JPEG; not exact browser resampling."""
    with Image.open(path) as image:
        rgb = image_array(image,size)
    output = io.BytesIO()
    Image.fromarray(rgb).save(output,format='JPEG',quality=quality)
    return output.getvalue()


def softmax(logits, temperature=1.):
    x=np.asarray(logits,dtype=np.float64)/float(temperature)
    x-=x.max(axis=-1,keepdims=True)
    e=np.exp(x)
    return e/e.sum(axis=-1,keepdims=True)


class TorchRuntime:
    def __init__(self,checkpoint,size=512,precision='fp32',channels_last=False,autotune=True,device='cuda'):
        if device=='cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable')
        self.device,self.size,self.half,self.channels = device,size,precision=='fp16',channels_last
        if self.half and device!='cuda': raise ValueError('FP16 CPU is not enabled')
        torch.backends.cudnn.benchmark=autotune
        self.model=TensorB1(checkpoint).eval().to(device)
        if self.half: self.model.half()
        if channels_last: self.model.to(memory_format=torch.channels_last)
        self.run(np.zeros((1,3,size,size),np.float32))

    @torch.inference_mode()
    def run(self,x):
        tensor=torch.from_numpy(x).to(self.device)
        if self.half: tensor=tensor.half()
        if self.channels: tensor=tensor.contiguous(memory_format=torch.channels_last)
        return self.model(tensor).float().cpu().numpy()


class ORTRuntime:
    def __init__(self,path,threads=4):
        import onnxruntime as ort
        options=ort.SessionOptions()
        options.intra_op_num_threads=threads
        options.inter_op_num_threads=1
        options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
        self.input=self.session.get_inputs()[0].name
    def run(self,x):
        return self.session.run(None,{self.input:x})[0]


def measure(runtime,items,repeats=100,warmup=20):
    """Same bytes for all configs; wall clock includes returning logits to CPU."""
    records=[]
    for i in range(warmup+repeats):
        raw=items[i%len(items)]
        start=time.perf_counter()
        x=decode(raw,runtime.size if hasattr(runtime,'size') else 512)
        before=time.perf_counter()
        logits=runtime.run(x)
        after=time.perf_counter()
        softmax(logits).argmax(1)
        end=time.perf_counter()
        if i>=warmup:
            records.append([(before-start)*1000,(after-before)*1000,(end-after)*1000,(end-start)*1000])
    times=np.asarray(records)
    return {f'{name}_{stat}_ms':float(np.quantile(times[:,j],q))
        for j,name in enumerate(['preprocess','infer_with_transfer','postprocess','server'])
        for stat,q in [('p50',.5),('p95',.95)]}
