"""Export error analysis cases for qualitative review."""
import argparse
import pandas as pd
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=str, required=True, help="Merged annotations CSV")
    parser.add_argument("--out_dir", type=str, default="outputs/tables")
    return parser.parse_args()

def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(args.annotations)
    
    # Needs valid gold
    df_valid = df.dropna(subset=['annotator_unsupported_label'])
    
    if len(df_valid) == 0:
        print("No valid annotations found.")
        return
        
    u_true = df_valid['annotator_unsupported_label'].astype(int)
    u_pred = df_valid['existing_judge_unsupported_label'].fillna(-1).astype(int)
    
    fn = df_valid[(u_true == 1) & (u_pred == 0)]
    fp = df_valid[(u_true == 0) & (u_pred == 1)]
    
    fn.to_csv(out_dir / "errors_unsupported_fn.csv", index=False)
    fp.to_csv(out_dir / "errors_unsupported_fp.csv", index=False)
    
    # Utility errors
    mask_u1 = (u_true == 1) & (u_pred == 1)
    df_util = df_valid[mask_u1].copy()
    if len(df_util) > 0:
        ut_true = df_util['annotator_utility_label'].astype(str)
        ut_pred = df_util['existing_judge_utility_label'].astype(str)
        
        util_errors = df_util[ut_true != ut_pred]
        util_errors.to_csv(out_dir / "errors_utility_mismatch.csv", index=False)
        
    print(f"Exported error cases to {out_dir}")

if __name__ == "__main__":
    main()
