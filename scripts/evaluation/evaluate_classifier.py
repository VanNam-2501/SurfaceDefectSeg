import sys
import argparse
import json
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import numpy as np

# Add src to python path
src_dir = Path(__file__).resolve().parent.parent.parent / "src"
sys.path.append(str(src_dir))

from straw_defect_cls.models import build_model
from straw_defect_cls.dataset import get_dataloaders
from straw_defect_cls.project_paths import get_default_config

def evaluate_model(model, dataloader, device, class_names):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    cm = confusion_matrix(all_labels, all_preds)
    
    return report, cm, all_preds, all_labels

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained classifier")
    parser.add_argument("--model", type=str, required=True, help="Model name (resnet18, efficientnet_b0, mobilenet_v3_small)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--dataset-root", type=str, help="Path to dataset root directory")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"], help="Dataset split to evaluate on")
    parser.add_argument("--output-dir", type=str, help="Path to save evaluation results")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    
    args = parser.parse_args()
    
    config = get_default_config()
    if args.dataset_root:
        config["dataset_root"] = args.dataset_root
        
    output_dir = Path(args.output_dir) if args.output_dir else Path(config["output_dir"]) / "eval" / args.model / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading {args.model} from {args.checkpoint}")
    model = build_model(args.model, num_classes=5)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model = model.to(device)
    
    # Need to load data
    dataloaders, class_names = get_dataloaders(
        data_dir=config["dataset_root"],
        batch_size=args.batch_size,
        num_workers=4
    )
    
    if args.split not in dataloaders:
        print(f"Split {args.split} not found in dataloaders.")
        return
        
    print(f"Evaluating on {args.split} split...")
    report, cm, _, _ = evaluate_model(model, dataloaders[args.split], device, class_names)
    
    # Save report
    with open(output_dir / "classification_report.json", "w") as f:
        json.dump(report, f, indent=4)
        
    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - {args.model} ({args.split})")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png")
    plt.close()
    
    print(f"Results saved to {output_dir}")
    print("Accuracy:", report["accuracy"])

if __name__ == "__main__":
    main()
