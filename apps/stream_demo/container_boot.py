"""Prepare a target-GPU engine, then run the complete demo in one container."""
import configparser
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

from serving_contract import DEFAULT_ARTIFACT, file_sha256, load_contract

RELEASE = 'v3.5'
MODEL = Path('/opt/straw/model')
ROOT = Path(__file__).resolve().parents[2]


def build_key(model, gpu_identity, trt_version):
    manifest = json.loads((model / 'manifest.json').read_text())
    onnx = model / f"b1_rawrgb{manifest['size']}.onnx"
    if file_sha256(onnx) != manifest['onnx_sha256']:
        raise ValueError('Packaged ONNX does not match the pinned manifest')
    return {'onnx_sha256': manifest['onnx_sha256'], 'gpu_identity': gpu_identity,
            'tensorrt_version': trt_version, 'release': RELEASE,
            'precision': 'fp16', 'workspace_mib': 1024}


def cached_engine_valid(out, key):
    try:
        previous = json.loads((out / 'target_engine.json').read_text())
        return previous['build_key'] == key and previous['engine_sha256'] == file_sha256(out / 'b1_rtx3050_fp16.engine')
    except (OSError, ValueError, KeyError, TypeError):
        return False


def prepare_engine(model=MODEL, root=ROOT):
    import tensorrt as trt
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,uuid', '--format=csv,noheader'], text=True).strip().splitlines()[0]
    key = build_key(model, gpu, trt.__version__)
    out = root / 'artifacts' / DEFAULT_ARTIFACT
    out.mkdir(parents=True, exist_ok=True)
    for name in ('manifest.json', 'nvinfer.txt', 'b1_rawrgb512.onnx'):
        target = out / name
        if not target.is_file() or file_sha256(target) != file_sha256(model / name):
            shutil.copyfile(model / name, target)
    engine = out / 'b1_rtx3050_fp16.engine'
    if not cached_engine_valid(out, key):
        print('Building TensorRT FP16 engine for the current GPU. The first start can take several minutes.', flush=True)
        temporary = engine.with_name('engine-' + uuid.uuid4().hex + '.tmp')
        try:
            subprocess.run(['/usr/src/tensorrt/bin/trtexec', '--onnx=' + str(out / 'b1_rawrgb512.onnx'),
                '--saveEngine=' + str(temporary), '--fp16', '--skipInference', '--memPoolSize=workspace:1024'], check=True)
            if not temporary.is_file() or temporary.stat().st_size == 0:
                raise RuntimeError('TensorRT did not create an engine')
            temporary.replace(engine)
        finally:
            temporary.unlink(missing_ok=True)
        (out / 'target_engine.json').write_text(json.dumps({'build_key': key, 'engine_sha256': file_sha256(engine)}, indent=2))
    else:
        print('Reusing the verified engine for this GPU/driver/TensorRT/model.', flush=True)
    config = configparser.ConfigParser(interpolation=None)
    config.read(out / 'nvinfer.txt')
    canonical = json.dumps({s: dict(config.items(s)) for s in config.sections()}, sort_keys=True).encode()
    manifest = json.loads((out / 'manifest.json').read_text())
    cache = {'context': {'checkpoint_sha256': manifest['checkpoint_sha256'], 'onnx_sha256': key['onnx_sha256'],
        'gpu_identity': gpu, 'sdk_image_id': os.environ['STRAW_BASE_IMAGE'],
        'inference_config_sha256': hashlib.sha256(canonical).hexdigest()},
        'engine_sha256': file_sha256(engine), 'build_key': key,
        'scope': 'Target-GPU TensorRT build; worker validates IO and warmup. No classification accuracy certification.'}
    (out / 'engine_cache.json').write_text(json.dumps(cache, indent=2))
    identity = load_contract(out)
    print('Model contract: ' + identity['contract_sha256'], flush=True)
    # A previous process cannot lend its fresh heartbeat to the new container.
    (out / 'resident_queue/runtime.json').unlink(missing_ok=True)


def main():
    import fcntl
    if os.environ.get('STRAW_RUNTIME') != 'native':
        raise RuntimeError('This entrypoint is for the self-contained Docker image')
    (ROOT / 'artifacts').mkdir(parents=True, exist_ok=True)
    with (ROOT / 'artifacts/demo.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another demo already owns this data volume') from None
        prepare_engine()
        from preview_web import main as web_main
        sys.argv = ['preview_web', '--host', '0.0.0.0', '--port', '8033', '--wsl-root', str(ROOT)]
        web_main()


if __name__ == '__main__':
    main()
