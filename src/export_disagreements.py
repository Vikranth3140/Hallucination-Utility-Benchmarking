"""Export disagreements from merged annotations."""
import argparse
import pandas as pd
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged", type=str, required=True, help="Merged annotations from merge_annotations.py")
    parser.add_argument("--output", type=str, required=True, help="Output CSV for disagreements")
    return parser.parse_args()

def main():
    args = parse_args()
    
    df = pd.read_csv(args.merged)
    
    # Disagreement when final tags are empty
    disagreements = df[df['final_unsupported_label'].isna() | ((df['final_unsupported_label'] == 1) & df['final_utility_label'].isna())]
    
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    disagreements.to_csv(args.output, index=False)
    print(f"Exported {len(disagreements)} disagreements for adjudication to {args.output}")

if __name__ == "__main__":
    main()
