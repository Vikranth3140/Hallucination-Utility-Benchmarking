"""Sample examples for human annotation."""
import argparse
import pandas as pd
import uuid
import json
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, default="data/labeled_PRESSURED.jsonl")
    parser.add_argument("--output", "-o", type=str, default="outputs/annotations/annotation_sample.csv")
    parser.add_argument("--n", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def main():
    args = parse_args()
    
    data = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
                
    if not data:
        print("No valid input data found.")
        return

    df = pd.DataFrame(data)
    
    # Default column assignments for backward compatibility handling
    if 'meta' not in df.columns:
        df['meta'] = [{}] * len(df)
    if 'unsupported_label' not in df.columns:
        df['unsupported_label'] = 0
    if 'utility_label' not in df.columns:
        df['utility_label'] = None
        
    df['generation_mode'] = df['meta'].apply(lambda x: x.get('mode', 'SAFE') if isinstance(x, dict) else 'SAFE')
    
    # Fill NA to ensure valid string concatenation
    df['task_type'] = df['task_type'].fillna('unknown')
    df['model_name'] = df['model_name'].fillna('unknown')
    
    df['stratify_key'] = df['task_type'] + "_" + df['model_name'] + "_" + df['generation_mode'] + "_" + df['unsupported_label'].astype(str) + "_" + df['utility_label'].astype(str)
    
    # We want to sample proportionally:
    if len(df) > args.n:
        sampled = df.groupby('stratify_key', group_keys=False).apply(
            lambda x: x.sample(n=max(1, int(len(x)/len(df) * args.n)), random_state=args.seed)
        )
        if len(sampled) < args.n:
            remaining = df.drop(sampled.index).sample(n=args.n - len(sampled), random_state=args.seed)
            sampled = pd.concat([sampled, remaining])
        elif len(sampled) > args.n:
            sampled = sampled.sample(n=args.n, random_state=args.seed)
    else:
        sampled = df

    sampled['sample_id'] = [str(uuid.uuid4()) for _ in range(len(sampled))]
    sampled['existing_judge_unsupported_label'] = sampled['unsupported_label']
    sampled['existing_judge_utility_label'] = sampled['utility_label']
    sampled['annotator_unsupported_label'] = ""
    sampled['annotator_utility_label'] = ""
    sampled['annotator_notes'] = ""

    cols = [
        "sample_id", "task_type", "prompt", "model_name", "generation_mode", "response",
        "existing_judge_unsupported_label", "existing_judge_utility_label",
        "annotator_unsupported_label", "annotator_utility_label", "annotator_notes"
    ]
    
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sampled[cols].to_csv(out_path, index=False)
    print(f"Exported {len(sampled)} samples to {out_path}")

if __name__ == "__main__":
    main()
