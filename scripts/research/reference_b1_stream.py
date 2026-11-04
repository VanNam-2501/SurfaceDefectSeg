"""Quality reference on the same H.264 clip, decoded by FFmpeg in isolated WSL.

No latency claim: pipe transfer and concurrent environment setup affect timing.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.straw_global_local.data import file_hash
from src.straw_global_local.deployment import TorchRuntime, normalize
from src.straw_global_local.runner import classification_metrics


def main():
    torch.set_num_threads(4)
    variant = os.environ.get('STRAW_STREAM_ARTIFACT', 'b1_stream')
    if variant not in ('b1_stream', 'b1_stream_512'): raise ValueError('Unsupported stream artifact directory')
    out = ROOT / 'artifacts' / variant
    manifest = json.loads((out/'manifest.json').read_text())
    cfg_file = 'straw_realtime_stream512.json' if variant == 'b1_stream_512' else 'straw_realtime.json'
    cfg = json.loads((ROOT/'configs'/cfg_file).read_text())
    if cfg['checkpoint_sha256'] != manifest['checkpoint_sha256']:
        raise RuntimeError('Serving config changed')
    runtime = TorchRuntime(ROOT/cfg['checkpoint'],size=cfg['size'],precision=cfg['precision'],channels_last=cfg['channels_last'])
    video = out/'quality.mp4'
    # Binary stdout goes straight through Popen, never PowerShell text decoding.
    linux_video=subprocess.check_output(['wsl','-d','TTTN-DeepStream','-u','root','--','wslpath','-u',video.as_posix()],text=True).strip()
    command = ['wsl','-d','TTTN-DeepStream','-u','root','--','ffmpeg','-nostdin','-v','error',
               '-i',linux_video,'-an','-f','rawvideo','-pix_fmt','rgb24','pipe:1']
    predictions = []
    with (out/'reference_decode.log').open('wb') as error:
        proc = subprocess.Popen(command,stdout=subprocess.PIPE,stderr=error)
        try:
            frame_bytes = 640*640*3
            while True:
                raw = bytearray()
                while len(raw)<frame_bytes:
                    block=proc.stdout.read(frame_bytes-len(raw))
                    if not block: break
                    raw.extend(block)
                if not raw: break
                if len(raw)!=frame_bytes: raise RuntimeError('Truncated decoded frame')
                rgb=np.frombuffer(raw,dtype=np.uint8).reshape(640,640,3)
                resized=np.asarray(Image.fromarray(rgb).resize((cfg['size'],cfg['size']),Image.Resampling.BILINEAR)).copy()
                predictions.append(runtime.run(normalize(resized))[0])
            code=proc.wait(timeout=20)
            if code: raise RuntimeError(f'FFmpeg failed: {code}')
        finally:
            if proc.poll() is None: proc.kill(); proc.wait()
            proc.stdout.close()
    if len(predictions)!=len(manifest['cases']): raise RuntimeError('Decoded frame count differs from manifest')
    labels=np.array([r['label'] for r in manifest['cases']])
    result=classification_metrics(labels,np.asarray(predictions))
    result.update(checkpoint_sha256=manifest['checkpoint_sha256'], video_sha256=file_hash(video),
                  note=f"Quality only. Same H.264 clip, FFmpeg CPU decode, PIL resize, CUDA {cfg['precision']} model. No latency claim.")
    (out/'torch_h264_reference.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    np.savez_compressed(out/'torch_h264_predictions.npz',logits=np.asarray(predictions),labels=labels)
    print('H264 reference:',result['macro_f1'],result['fn'],result['fp'],flush=True)


if __name__=='__main__': main()
