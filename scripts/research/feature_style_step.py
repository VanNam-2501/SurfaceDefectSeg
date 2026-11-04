"""One real-image forward/backward step, NOT a model-quality experiment."""
from pathlib import Path
import argparse
import json
import sys
import time

import pandas as pd
from PIL import Image, ImageOps
import torch
from torchvision.transforms.functional import to_tensor, normalize

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.inspection_research.feature_style import StyleResNet18, paired_training_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-csv', type=Path, default=ROOT / 'artifacts/straw_thesis/data_v2/iid/train.csv')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/verification/feature_style_step.json')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    torch.manual_seed(42)
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; pass --device cpu explicitly')
    frame = pd.read_csv(args.train_csv)
    if not {'image_path', 'label'}.issubset(frame.columns) or frame.label.min() != 0:
        raise ValueError('Require image_path and integer labels starting at zero')
    classes = sorted(frame.label.unique())
    if classes != list(range(len(classes))):
        raise ValueError('Labels must be contiguous 0..C-1')
    # Four images, two different classes with two instances each, to fit a 4GB GPU.
    subset = pd.concat([frame[frame.label == label].head(2) for label in classes[:2]])
    if len(subset) != 4:
        raise ValueError('Need at least two training images in each of two classes')
    images = []
    for path in subset.image_path:
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert('RGB').resize((224, 224), Image.Resampling.BILINEAR)
        images.append(normalize(to_tensor(image), [.485, .456, .406], [.229, .224, .225]))
    batch = torch.stack(images).to(args.device)
    labels = torch.tensor(subset.label.to_numpy(), dtype=torch.long, device=args.device)
    model = StyleResNet18(len(classes), pretrained=False, probability=1, strength=.5).to(args.device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    before = model.network.conv1.weight.detach().clone()
    start = time.perf_counter()
    clean = model(batch)
    mixed = model(batch, labels=labels, augment=True, pairing='same_class')
    loss, metrics = paired_training_loss(clean, mixed, labels, consistency_weight=.2)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient_norm = float(model.network.conv1.weight.grad.norm().detach())
    optimizer.step()
    if args.device.startswith('cuda'):
        torch.cuda.synchronize()
    change = float((model.network.conv1.weight.detach() - before).abs().max())
    if not torch.isfinite(loss) or gradient_norm <= 0 or change <= 0:
        raise RuntimeError('Nonfinite loss or no encoder update')
    model.eval()
    with torch.no_grad():
        inference = model(batch)  # Labels and domains deliberately omitted.
    evidence = {
        'status': 'TECHNICAL STEP ONLY, NOT ACCURACY OR GENERALIZATION RESULTS',
        'pretrained': False, 'device': args.device, 'batch_size': len(batch), 'num_classes': len(classes),
        'source_manifest': str(args.train_csv), 'training_steps': 1,
        'loss': {name: float(value) for name, value in metrics.items()},
        'encoder_gradient_norm': gradient_norm, 'encoder_max_weight_change': change,
        'inference_without_labels_shape': list(inference.shape), 'seconds': time.perf_counter() - start,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2), encoding='utf-8')
    print(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    main()
