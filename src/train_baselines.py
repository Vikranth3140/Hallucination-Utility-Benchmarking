"""Train diagnostic baselines for hallucination detection."""
import argparse
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score, f1_score

logging.basicConfig(level=logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default="outputs/data/train.jsonl")
    parser.add_argument("--test", type=str, default="outputs/data/test.jsonl")
    parser.add_argument("--target", type=str, choices=["unsupported_label", "utility_label"], default="unsupported_label")
    parser.add_argument("--ablation", type=str, choices=["majority", "task", "prompt", "output", "task_prompt_output"], default="task_prompt_output")
    parser.add_argument("--out_dir", type=str, default="outputs/models")
    return parser.parse_args()

def load_data(file_path, target):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    df = pd.DataFrame(data)
    
    if target == "utility_label":
        # Only evaluate on examples where unsupported_label == 1
        df = df[df['unsupported_label'] == 1].copy()
        
    return df

def build_model(ablation, target, class_weight='balanced'):
    if ablation == "majority":
        return DummyClassifier(strategy="most_frequent")
        
    transformers = []
    if ablation in ["task", "task_prompt_output"]:
        transformers.append(('task', OneHotEncoder(handle_unknown='ignore'), ['task_type']))
    if ablation in ["prompt", "task_prompt_output"]:
        transformers.append(('prompt', TfidfVectorizer(max_features=1000), 'prompt'))
    if ablation in ["output", "task_prompt_output"]:
        transformers.append(('output', TfidfVectorizer(max_features=3000), 'response'))
        
    preprocessor = ColumnTransformer(transformers=transformers)
    
    # Use logistic regression
    clf = LogisticRegression(class_weight=class_weight, max_iter=1000, random_state=42)
    
    return Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', clf)
    ])

def main():
    args = parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    
    train_df = load_data(args.train, args.target)
    test_df = load_data(args.test, args.target)
    
    if len(train_df) == 0 or len(test_df) == 0:
        logging.error("No data available for the given target.")
        return
        
    y_train = train_df[args.target].fillna('None' if args.target == "utility_label" else 0).astype(str)
    y_test = test_df[args.target].fillna('None' if args.target == "utility_label" else 0).astype(str)
    
    model = build_model(args.ablation, args.target)
    
    logging.info(f"Training {args.ablation} baseline on {len(train_df)} examples...")
    model.fit(train_df, y_train)
    
    logging.info(f"Evaluating on {len(test_df)} examples...")
    p_test = model.predict(test_df)
    
    rep = classification_report(y_test, p_test, zero_division=0)
    print("="*50)
    print(f"Target: {args.target} | Ablation: {args.ablation}")
    print("="*50)
    print(rep)
    
    # Save metrics
    metrics = {
        "accuracy": accuracy_score(y_test, p_test),
        "macro_f1": f1_score(y_test, p_test, average='macro', zero_division=0),
        "weighted_f1": f1_score(y_test, p_test, average='weighted', zero_division=0)
    }
    
    out_file = Path(args.out_dir) / f"{args.target}_{args.ablation}_metrics.json"
    with open(out_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {out_file}")

if __name__ == "__main__":
    main()
