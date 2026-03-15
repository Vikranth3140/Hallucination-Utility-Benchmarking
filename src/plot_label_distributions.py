"""Plot label distributions generated from compute_dataset_stats."""
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables_dir", type=str, default="outputs/tables")
    parser.add_argument("--out_dir", type=str, default="outputs/figures")
    return parser.parse_args()

def main():
    args = parse_args()
    tables_dir = Path(args.tables_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot unsupported by task
    path_unsup_task = tables_dir / "unsupported_by_task.csv"
    if path_unsup_task.exists():
        df = pd.read_csv(path_unsup_task)
        plt.figure(figsize=(6, 4))
        plt.bar(df['task_type'], df['unsupported_label'], color='skyblue')
        plt.title('Unsupported Content Rate by Task Type')
        plt.ylabel('Rate')
        plt.ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig(out_dir / "unsupported_by_task.png", dpi=300)
        plt.close()
        
    # Plot utility by task
    path_util_task = tables_dir / "utility_by_task.csv"
    if path_util_task.exists():
        df = pd.read_csv(path_util_task)
        df.set_index('task_type').plot(kind='bar', stacked=True, figsize=(8, 5), colormap='viridis')
        plt.title('Utility Distribution given Unsupported=1')
        plt.ylabel('Proportion')
        plt.legend(title='Utility Label', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(out_dir / "utility_by_task_stacked.png", dpi=300)
        plt.close()

    print(f"Exported plots to {out_dir}")

if __name__ == "__main__":
    main()
