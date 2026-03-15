"""Prompt management utilities."""
import json
import hashlib
from pathlib import Path
from typing import List, Tuple
from .schema import PromptItem, TaskType

def load_prompts(path: str) -> List[PromptItem]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [PromptItem(**x) for x in raw]

def get_template(name: str) -> str:
    path = Path("prompts") / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Missing prompt template: {path}")
    return path.read_text(encoding="utf-8")

def get_template_hash(template_text: str) -> str:
    return hashlib.md5(template_text.encode("utf-8")).hexdigest()[:8]

def build_generation_prompt(task_type: TaskType, prompt: str, mode: str) -> Tuple[str, str]:
    mode_lower = mode.lower()
    if mode_lower not in ["safe", "pressured"]:
        mode_lower = "safe" # fallback
        
    template_name = f"generation_{mode_lower}"
    template = get_template(template_name)
    filled_prompt = template.format(task_type=task_type, prompt=prompt)
    return filled_prompt, get_template_hash(template)

def build_judge_unsupported_prompt(task_type: TaskType, prompt: str, output: str) -> Tuple[str, str]:
    template = get_template("judge_unsupported")
    filled_prompt = template.format(task_type=task_type, prompt=prompt, output=output)
    return filled_prompt, get_template_hash(template)

def build_judge_utility_prompt(task_type: TaskType, prompt: str, output: str) -> Tuple[str, str]:
    template = get_template("judge_utility")
    filled_prompt = template.format(task_type=task_type, prompt=prompt, output=output)
    return filled_prompt, get_template_hash(template)

def log_prompt_used(log_file: str, prompt_data: dict):
    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(prompt_data) + "\n")
