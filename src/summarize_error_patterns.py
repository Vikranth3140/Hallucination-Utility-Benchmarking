"""Summarize error cases into markdown reports."""
import pandas as pd
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--errors_dir", type=str, default="outputs/tables")
    parser.add_argument("--out_dir", type=str, default="outputs/tables")
    args = parser.parse_args()
    
    errors_dir = Path(args.errors_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    md_lines = ["# Error Analysis Report\n"]
    
    fn_file = errors_dir / "errors_unsupported_fn.csv"
    if fn_file.exists():
        df_fn = pd.read_csv(fn_file)
        md_lines.append(f"## False Negatives (Judge=0, Human=1): {len(df_fn)}\n")
        
    fp_file = errors_dir / "errors_unsupported_fp.csv"
    if fp_file.exists():
        df_fp = pd.read_csv(fp_file)
        md_lines.append(f"## False Positives (Judge=1, Human=0): {len(df_fp)}\n")
        
    util_file = errors_dir / "errors_utility_mismatch.csv"
    if util_file.exists():
        df_util = pd.read_csv(util_file)
        md_lines.append(f"## Utility Mismatches: {len(df_util)}\n")
        for idx, row in df_util.head(10).iterrows():
            md_lines.append(f"**Prompt:** {row['prompt']}")
            md_lines.append(f"**Response:** {row['response']}")
            md_lines.append(f"**Human Utility:** {row['annotator_utility_label']} | **Judge Utility:** {row['existing_judge_utility_label']}\n")
            md_lines.append("---\n")
            
    with open(out_dir / "error_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
        
    print(f"Exported markdown summary to {out_dir / 'error_summary.md'}")

if __name__ == "__main__":
    main()
