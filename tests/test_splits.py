import pytest
import pandas as pd
import json
from pathlib import Path
import subprocess

def test_splits_no_leakage(tmp_path):
    data = [
        {"prompt_id": "p1", "task_type": "factual", "unsupported_label": 1, "utility_label": "useful", "sample_id": "s1"},
        {"prompt_id": "p1", "task_type": "factual", "unsupported_label": 1, "utility_label": "useful", "sample_id": "s2"},
        {"prompt_id": "p2", "task_type": "creative", "unsupported_label": 0, "utility_label": None, "sample_id": "s3"},
        {"prompt_id": "p3", "task_type": "brainstorm", "unsupported_label": 1, "utility_label": "harmful", "sample_id": "s4"},
        {"prompt_id": "p4", "task_type": "factual", "unsupported_label": 0, "utility_label": None, "sample_id": "s5"}
    ]
    inp_file = tmp_path / "input.jsonl"
    with open(inp_file, "w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")
            
    out_dir = tmp_path / "outputs"
    
    cmd = ["python", "-m", "src.create_splits", "--input", str(inp_file), "--out_dir", str(out_dir), "--folds", "2"]
    subprocess.run(cmd, check=True)
    
    manifest_path = out_dir / "split_manifest.json"
    assert manifest_path.exists()
    with open(manifest_path, "r") as f:
        splits = json.load(f)
        
    for split in splits:
        train_ids = split["train_indices"]
        val_ids = split["val_indices"]
        
        sid_to_pid = {d["sample_id"]: d["prompt_id"] for d in data}
        
        train_pids = set(sid_to_pid[sid] for sid in train_ids)
        val_pids = set(sid_to_pid[sid] for sid in val_ids)
        
        # Intersection must be empty
        assert len(train_pids.intersection(val_pids)) == 0
