"""Freeze a separate RTSP experiment: raw-RGB ONNX with embedded normalization.

Uses only the existing validation set. Exports CPU ONNX parity evidence before
any DeepStream/TensorRT conversion. No original checkpoints/configs are changed.
"""
import json
import os
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.straw_global_local.deployment import TensorB1, normalize
from src.straw_global_local.data import CLASSES, file_hash


class RawRGBB1(nn.Module):
    def __init__(self, checkpoint):
        super().__init__()
        self.base = TensorB1(checkpoint).float().eval()
        self.register_buffer('mean', torch.tensor([.485, .456, .406]).reshape(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([.229, .224, .225]).reshape(1, 3, 1, 1))

    def forward(self, rgb):
        # nvinfer has one net-scale-factor; ImageNet uses different channel std.
        # Embedding the exact normalization prevents silently using one std.
        return self.base((rgb / 255. - self.mean) / self.std).softmax(dim=1)


def main():
    import onnx
    import onnxruntime as ort
    variant = os.environ.get('STRAW_STREAM_ARTIFACT', 'b1_stream')
    if variant not in ('b1_stream', 'b1_stream_512', 'b1_stream_512_pose_v1'):
        raise ValueError('Unsupported stream artifact directory')
    out = ROOT / 'artifacts' / variant
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    cfg_file = ('straw_realtime_stream512_pose_v1.json' if variant == 'b1_stream_512_pose_v1'
                else 'straw_realtime_stream512.json' if variant == 'b1_stream_512'
                else 'straw_realtime.json')
    cfg = json.loads((ROOT / 'configs' / cfg_file).read_text())
    size = cfg['size']
    policy = None
    if cfg.get('decision_policy'):
        policy = json.loads((ROOT/cfg['decision_policy']).read_text(encoding='utf-8'))
        if (policy.get('checkpoint_sha256') != cfg['checkpoint_sha256'] or
            policy.get('input_size') != size or policy.get('temperature') != cfg['temperature']):
            raise RuntimeError('Stream decision policy does not match checkpoint/preprocessing')
    checkpoint = ROOT / cfg['checkpoint']
    if file_hash(checkpoint) != cfg['checkpoint_sha256']:
        raise RuntimeError('Serving checkpoint hash mismatch')
    source = ROOT / 'artifacts/b1_deployment/frame640'
    index = json.loads((source / 'index.json').read_text())
    if index['manifest_sha256'] != file_hash(ROOT / 'artifacts/straw_global_local/data_v1/val.csv'):
        raise RuntimeError('Validation manifest changed')
    cases = index['cases']
    frames = out / 'frames'
    frames.mkdir(exist_ok=True)
    from shutil import copyfile
    for i, case in enumerate(cases):
        path = source / f"{case['sample_id']}.jpg"
        if file_hash(path) != case['sha256']:
            raise RuntimeError(f'Proxy image hash mismatch: {path}')
        copyfile(path, frames / f'{i:04d}.jpg')
    model = RawRGBB1(checkpoint).eval()
    target = out / f'b1_rawrgb{size}.onnx'
    torch.onnx.export(model, torch.zeros(1, 3, size, size), str(target),
                      input_names=['rgb'], output_names=['probabilities'], opset_version=17, dynamo=False)
    onnx.checker.check_model(onnx.load(str(target)))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(target), sess_options=options, providers=['CPUExecutionProvider'])
    errors, reference_errors, changed = [], [], 0
    for case in cases[::15]:
        with Image.open(source / f"{case['sample_id']}.jpg") as im:
            rgb = np.asarray(im.convert('RGB').resize((size, size), Image.Resampling.BILINEAR)).copy()
        x = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32)
        with torch.inference_mode():
            expected = model.base(torch.from_numpy(normalize(rgb))).softmax(dim=1).numpy()
            wrapped = model(torch.from_numpy(x)).numpy()
        actual = session.run(None, {'rgb': x})[0]
        reference_errors.append(float(np.max(np.abs(expected - wrapped))))
        errors.append(float(np.max(np.abs(expected - actual))))
        changed += int(expected.argmax() != actual.argmax())
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-4)
    manifest = dict(schema=1, role='val', source='static photos, not motion/camera data',
                    classes=CLASSES, size=size, stream_width=640, stream_height=640,
                    checkpoint_sha256=cfg['checkpoint_sha256'], onnx_sha256=file_hash(target),
                    decision_threshold=policy['threshold'] if policy else None,
                    validation_manifest_sha256=index['manifest_sha256'],
                    cases=cases, parity=dict(n=len(errors), max_probability_abs_error=max(errors),
                    embedded_normalize_error=max(reference_errors), changed_predictions=changed),
                    preprocessing='RGB float 0..255 NCHW; normalization embedded in ONNX; square resize')
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    config = f'''[property]
gpu-id=0
model-engine-file=/workspace/artifacts/b1_stream/b1_rtx3050_fp16.engine
batch-size=1
network-mode=2
network-type=1
classifier-threshold=0.0
process-mode=1
gie-unique-id=1
interval=0
infer-dims=3;{size};{size}
model-color-format=0
net-scale-factor=1.0
offsets=0;0;0
maintain-aspect-ratio=0
scaling-filter=1
scaling-compute-hw=1
output-tensor-meta=1
output-blob-names=probabilities
output-io-formats=probabilities:fp32:chw
workspace-size=512
'''
    (out / 'nvinfer.txt').write_text(config, encoding='utf-8')
    print(json.dumps(manifest['parity']), flush=True)


if __name__ == '__main__':
    main()
