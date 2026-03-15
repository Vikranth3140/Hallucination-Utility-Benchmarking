"""Merge annotations to support adjudication."""
import argparse
import pandas as pd
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a1", type=str, required=True, help="Annotator 1 CSV")
    parser.add_argument("--a2", type=str, required=True, help="Annotator 2 CSV")
    parser.add_argument("--output", type=str, required=True, help="Merged CSV output")
    return parser.parse_args()

def main():
    args = parse_args()
    
    df1 = pd.read_csv(args.a1)
    df2 = pd.read_csv(args.a2)
    
    df = pd.merge(df1, df2[['sample_id', 'annotator_unsupported_label', 'annotator_utility_label', 'annotator_notes']], 
                  on="sample_id", suffixes=('_a1', '_a2'))
    
    # Auto-adjudicate where they agree
    def resolve_unsupported(row):
        if row['annotator_unsupported_label_a1'] == row['annotator_unsupported_label_a2']:
            return row['annotator_unsupported_label_a1']
        return None
        
    def resolve_utility(row):
        # only matters if unsupported was 1 for both
        if row['annotator_unsupported_label_a1'] == 1 and row['annotator_unsupported_label_a2'] == 1:
            if row['annotator_utility_label_a1'] == row['annotator_utility_label_a2']:
                return row['annotator_utility_label_a1']
        return None

    df['final_unsupported_label'] = df.apply(resolve_unsupported, axis=1)
    df['final_utility_label'] = df.apply(resolve_utility, axis=1)
    
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Saved merged annotations to {args.output}")

if __name__ == "__main__":
    main()
