"""Stage only verified model/demo assets into the allowlisted Docker context."""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from apps.stream_demo.serving_contract import DEFAULT_ARTIFACT, file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, default=ROOT / 'artifacts' / DEFAULT_ARTIFACT)
    parser.add_argument('--samples-dir', type=Path, default=ROOT / 'demo_samples')
    args = parser.parse_args()
    pinned = json.loads((ROOT / 'docker/demo/model/manifest.json').read_text())
    model = args.model_dir / 'b1_rawrgb512.onnx'
    if not model.is_file() or file_sha256(model) != pinned['onnx_sha256']:
        raise ValueError('Supply the pinned b1_rawrgb512.onnx from the handover/model bundle with --model-dir')
    target = ROOT / '.docker/model'
    target.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(model, target / model.name)
    source_manifest = args.samples_dir / 'manifest.json'
    manifest = json.loads(source_manifest.read_text())
    samples = ROOT / '.docker/samples/demo_samples'
    samples.mkdir(parents=True, exist_ok=True)
    for item in manifest['cases']:
        relative = Path(item['path'])
        if relative.is_absolute() or '..' in relative.parts or relative.parts[0] != 'demo_samples':
            raise ValueError('Unsafe sample path')
        source = args.samples_dir.joinpath(*relative.parts[1:])
        if source.is_symlink() or file_sha256(source) != item['sha256']:
            raise ValueError('Demo sample differs from the validation manifest')
        dest = samples.joinpath(*relative.parts[1:])
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
    shutil.copyfile(source_manifest, samples / 'manifest.json')
    shutil.copyfile(args.samples_dir / 'conveyor_replay.mp4', samples / 'conveyor_replay.mp4')
    # A small context avoids walking datasets, virtual environments and research
    # caches in a developer checkout, even when Docker uses the legacy builder.
    context = ROOT / '.docker/context'
    def stage(source, relative):
        dest = context / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
    for source in (ROOT / 'apps/stream_demo').glob('*'):
        if source.is_file() and source.suffix in {'.py', '.js', '.html', '.css', '.cpp'} and not source.name.startswith('research'):
            stage(source, source.relative_to(ROOT))
    for relative in ('.dockerignore', 'docker/demo/Dockerfile', 'docker/demo/model/manifest.json',
                     'docker/demo/model/nvinfer.txt', 'configs/straw_realtime_stream512_pose_v1.json',
                     'requirements/deepstream-demo.txt'):
        stage(ROOT / relative, relative)
    stage(target / model.name, '.docker/model/' + model.name)
    for item in manifest['cases']:
        relative = Path(item['path'])
        stage(samples.joinpath(*relative.parts[1:]), Path('.docker/samples') / relative)
    for name in ('manifest.json', 'conveyor_replay.mp4'):
        stage(samples / name, '.docker/samples/demo_samples/' + name)
    print('Docker context: .docker/context (pinned ONNX, five validation photos and replay; no uploads/history/datasets).')


if __name__ == '__main__':
    main()
