import os
import argparse
import random
import shutil
from pathlib import Path
import pandas as pd

def split_dataset(data_dir, output_dir, train_ratio=0.7, val_ratio=0.15):
    data_path = Path(data_dir)
    out_path = Path(output_dir)
    
    classes = [d.name for d in data_path.iterdir() if d.is_dir()]
    print(f"Found classes: {classes}")
    
    all_data = []
    
    for cls in classes:
        cls_dir = data_path / cls
        images = [f.name for f in cls_dir.glob("*.jpg") if f.is_file()]
        # Try other extensions if needed
        if not images:
            images = [f.name for f in cls_dir.glob("*.png") if f.is_file()]
            
        random.shuffle(images)
        
        n = len(images)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train+n_val]
        test_imgs = images[n_train+n_val:]
        
        for img, split in zip([train_imgs, val_imgs, test_imgs], ['train', 'val', 'test']):
            for img_name in img:
                all_data.append({"filename": img_name, "class": cls, "split": split})
                
                # Copy file to split directory
                src = cls_dir / img_name
                dst_dir = out_path / split / cls
                dst_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst_dir / img_name)
                
    df = pd.DataFrame(all_data)
    
    print("\nDataset Statistics:")
    print(df.groupby(['split', 'class']).size().unstack(fill_value=0))
    
    df.to_csv(out_path / "dataset_splits.csv", index=False)
    print(f"\nSaved split information to {out_path / 'dataset_splits.csv'}")

def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for Straw Defect Classification")
    parser.add_argument("--source", type=str, required=True, help="Path to raw dataset directory")
    parser.add_argument("--output", type=str, required=True, help="Path to output split dataset directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    split_dataset(args.source, args.output)

if __name__ == "__main__":
    main()
