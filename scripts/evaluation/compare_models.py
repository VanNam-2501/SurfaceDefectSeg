import argparse
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Compare evaluation results of multiple models")
    parser.add_argument("--results-dir", type=str, required=True, help="Directory containing eval results (e.g., outputs/eval)")
    parser.add_argument("--output-dir", type=str, default="outputs/comparison", help="Directory to save comparison results")
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    data = []
    
    for model_dir in results_dir.iterdir():
        if model_dir.is_dir():
            model_name = model_dir.name
            for split in ["val", "test"]:
                report_path = model_dir / split / "classification_report.json"
                if report_path.exists():
                    with open(report_path, "r") as f:
                        report = json.load(f)
                        data.append({
                            "Model": model_name,
                            "Split": split,
                            "Accuracy": report["accuracy"],
                            "Macro F1": report["macro avg"]["f1-score"]
                        })
                        
    if not data:
        print("No evaluation results found.")
        return
        
    df = pd.DataFrame(data)
    
    print("Model Comparison:")
    print(df.to_string(index=False))
    df.to_csv(output_dir / "model_comparison.csv", index=False)
    
    # Plotting
    for split in ["val", "test"]:
        split_df = df[df["Split"] == split]
        if not split_df.empty:
            plt.figure(figsize=(10, 6))
            x = range(len(split_df))
            plt.bar(x, split_df["Accuracy"], width=0.4, label='Accuracy', align='center')
            plt.bar([p + 0.4 for p in x], split_df["Macro F1"], width=0.4, label='Macro F1', align='center')
            plt.xticks([p + 0.2 for p in x], split_df["Model"])
            plt.title(f"Model Comparison - {split.capitalize()} Set")
            plt.ylabel("Score")
            plt.legend()
            plt.ylim(0, 1.0)
            plt.tight_layout()
            plt.savefig(output_dir / f"comparison_chart_{split}.png")
            plt.close()
            
    print(f"Comparison results and charts saved to {output_dir}")

if __name__ == "__main__":
    main()
