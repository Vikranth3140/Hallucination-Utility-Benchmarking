"""Plot confusion matrices for judge vs human agreement."""
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged_csv", type=str, required=True, help="Merged annotations or sample annotations CSV")
    parser.add_argument("--out_dir", type=str, default="outputs/figures")
    return parser.parse_args()

def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(args.merged_csv)
    
    u_true = df['annotator_unsupported_label'].fillna(-1).astype(int)
    u_pred = df['existing_judge_unsupported_label'].fillna(-1).astype(int)
    
    # clean out valid comparisons
    mask = (u_true != -1) & (u_pred != -1)
    if mask.sum() > 0:
        cm = confusion_matrix(u_true[mask], u_pred[mask], labels=[0, 1])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
        disp.plot(cmap='Blues')
        plt.title('Judge vs Human: Unsupported Label')
        plt.tight_layout()
        plt.savefig(out_dir / "cm_unsupported_judge_vs_human.png", dpi=300)
        plt.close()

    mask_u1 = (u_true == 1) & (u_pred == 1)
    if mask_u1.sum() > 0:
        ut_true = df.loc[mask_u1, 'annotator_utility_label'].fillna('None').astype(str)
        ut_pred = df.loc[mask_u1, 'existing_judge_utility_label'].fillna('None').astype(str)
        
        labels = ['useful', 'neutral', 'harmful']
        cm = confusion_matrix(ut_true, ut_pred, labels=labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap='Oranges')
        plt.title('Judge vs Human: Utility Label')
        plt.tight_layout()
        plt.savefig(out_dir / "cm_utility_judge_vs_human.png", dpi=300)
        plt.close()

    print(f"Exported confusion matrices to {out_dir}")

if __name__ == "__main__":
    main()
