"""Small validation-free kernel timing probe; never used to measure accuracy."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from src.straw_global_local.model import GlobalLocalCNN

torch.set_num_threads(4)
checkpoint = torch.load('artifacts/local_b1_check/B1_s42/best.pt',map_location='cpu',weights_only=False)
for half, channels, autotune in [(False,False,False),(True,False,False),(True,True,True)]:
    torch.backends.cudnn.benchmark=autotune
    model=GlobalLocalCNN('B1',pretrained=False).eval().cuda()
    model.load_state_dict(checkpoint['model'])
    x=torch.randn(1,3,512,512,device='cuda')
    if half: model.half(); x=x.half()
    if channels: model.to(memory_format=torch.channels_last); x=x.contiguous(memory_format=torch.channels_last)
    times=[]
    with torch.inference_mode():
        for i in range(30):
            torch.cuda.synchronize(); start=time.perf_counter()
            model({'global':x}); torch.cuda.synchronize()
            if i>=10: times.append((time.perf_counter()-start)*1000)
    print(half, channels, autotune, sorted(times)[len(times)//2], flush=True)
    del model, x
