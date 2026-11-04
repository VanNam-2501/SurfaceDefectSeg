"""Infer one original image with a checkpoint-bound B1 512 decision policy."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.straw_global_local.data import CLASSES
from src.straw_global_local.deployment import TorchRuntime, decode
from src.straw_global_local.postprocess import apply_policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image', type=Path)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    policy = json.loads(args.policy.read_text(encoding='utf-8'))
    runtime = TorchRuntime(args.checkpoint, size=512, precision='fp32', device=args.device)
    logits = runtime.run(decode(args.image.read_bytes(), 512))
    predictions, p = apply_policy(logits, policy, args.checkpoint)
    index = int(predictions[0])
    print(json.dumps({
        'class': CLASSES[index], 'defective': bool(index != 0),
        'raw_argmax': CLASSES[int(p[0].argmax())],
        'defect_score': float(1-p[0,0]),
        'threshold': policy['threshold'],
        'probabilities': dict(zip(CLASSES, [float(v) for v in p[0]])),
    }, indent=2))


if __name__ == '__main__':
    main()
