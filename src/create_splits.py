"""Create and save reproducible data splits."""
import argparse
import pandas as pd
import json
from sklearn.model_selection import StratifiedGroupKFold
from pathlib import Path
import uuid

def parse_args():
    parser = argparse.ArgumentParser(description="Create dataset splits preventing prompt leakage.")
    parser.add_argument("--input", type=str, default="data/labeled_PRESSURED.jsonl")
    parser.add_argument("--out_dir", type=str, default="outputs/data")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
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
        print("No data found.")
        return
        
    df = pd.DataFrame(data)
    
    if 'sample_id' not in df.columns:
        df['sample_id'] = [str(uuid.uuid4()) for _ in range(len(df))]
    
    df['unsupported_label'] = df['unsupported_label'].fillna(0).astype(int)
    df['utility_label'] = df['utility_label'].fillna('None').astype(str)
    
    sgkf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    
    y_strat = df['task_type'].astype(str) + "_" + df['unsupported_label'].astype(str) + "_" + df['utility_label']
    
    splits = []
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(df, y_strat, groups=df['prompt_id'])):
        train_ids = df.iloc[train_idx]['sample_id'].tolist()
        val_ids = df.iloc[val_idx]['sample_id'].tolist()
        splits.append({
            "fold": fold,
            "train_indices": train_ids,
            "val_indices": val_ids
        })
        
    manifest_path = out_dir / "split_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)
        
    train_split = df[df['sample_id'].isin(splits[0]['train_indices'])]
    test_split = df[df['sample_id'].isin(splits[0]['val_indices'])]
    
    train_split.to_json(out_dir / "train.jsonl", orient="records", lines=True)
    test_split.to_json(out_dir / "test.jsonl", orient="records", lines=True)
    
    print(f"Created {args.folds}-fold splits and saved to {out_dir}")

if __name__ == "__main__":
    main()
