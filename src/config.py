"""Configuration loader for the project."""
import yaml
from pathlib import Path

def load_config(path="config.yaml"):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Warning: Could not load {path}. Using defaults. Error: {e}")
        return {}
