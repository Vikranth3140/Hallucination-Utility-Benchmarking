"""Compute dataset statistics and export tables."""
import argparse
import pandas as pd
import json
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/labeled_PRESSURED.jsonl")
    parser.add_argument("--out_dir", type=str, default="outputs/tables")
    return parser.parse_args()

def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    data = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
                
    if not data:
        print("No data to process.")
        return

    df = pd.DataFrame(data)
    df['generation_mode'] = df['meta'].apply(lambda x: x.get('mode', 'SAFE') if isinstance(x, dict) else 'SAFE')
    
    # 1. Unsupported rate by task type
    unsup_task = df.groupby('task_type')['unsupported_label'].mean().reset_index()
    unsup_task.to_csv(out_dir / "unsupported_by_task.csv", index=False)
    
    # 2. Unsupported rate by model
    unsup_model = df.groupby('model_name')['unsupported_label'].mean().reset_index()
    unsup_model.to_csv(out_dir / "unsupported_by_model.csv", index=False)
    
    # 3. Unsupported rate by mode
    unsup_mode = df.groupby('generation_mode')['unsupported_label'].mean().reset_index()
    unsup_mode.to_csv(out_dir / "unsupported_by_mode.csv", index=False)
    
    # 4. Utility distribution conditional on unsupported = 1
    df_u1 = df[df['unsupported_label'] == 1].copy()
    if len(df_u1) > 0:
        util_task = pd.crosstab(df_u1['task_type'], df_u1['utility_label'], normalize='index').reset_index()
        util_task.to_csv(out_dir / "utility_by_task.csv", index=False)
        
        util_model = pd.crosstab(df_u1['model_name'], df_u1['utility_label'], normalize='index').reset_index()
        util_model.to_csv(out_dir / "utility_by_model.csv", index=False)
        
        util_mode = pd.crosstab(df_u1['generation_mode'], df_u1['utility_label'], normalize='index').reset_index()
        util_mode.to_csv(out_dir / "utility_by_mode.csv", index=False)
    
    # Cross-tabs
    xtab_task_unsup = pd.crosstab(df['task_type'], df['unsupported_label'])
    xtab_task_unsup.to_csv(out_dir / "xtab_task_vs_unsupported.csv")
    
    print(f"Exported statistical tables to {out_dir}")

if __name__ == "__main__":
    main()
