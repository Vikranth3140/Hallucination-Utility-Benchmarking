"""Annotation utilities for labeling hallucinations."""
import json
import re
from tqdm import tqdm
from dotenv import load_dotenv
from .schema import ModelOutput, LabeledExample
from .prompts import build_judge_prompt
from .llm_clients import NIMClient

# Load environment variables
load_dotenv()


def safe_parse_json(s: str):
    """Best-effort: extract first JSON object."""
    if not s:
        return None
    s = s.strip()

    # direct parse
    try:
        return json.loads(s)
    except Exception:
        pass

    # find first {...}
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

def normalize_label(label: str) -> str:
    label = (label or "").strip().upper()
    if label in {"U+", "U0", "U-"}:
        return label
    if "U+" in label:
        return "U+"
    if "U-" in label:
        return "U-"
    return "U0"

def main():
    in_path = "data/raw_outputs.jsonl"
    out_path = "data/labeled.jsonl"

    judge_model_name = "nim-judge"
    judge = NIMClient(
        model="meta/llama-3.1-70b-instruct",
        temperature=0.0,      # CRITICAL - keep deterministic
        max_tokens=256,
    )

    with open(in_path, "r", encoding="utf-8") as f:
        content = f.read()
        # Split by double newline (entries are separated by \n\n)
        json_strings = [s.strip() for s in content.split('\n\n') if s.strip()]
        rows = [ModelOutput.model_validate_json(js) for js in json_strings]

    with open(out_path, "w", encoding="utf-8") as out:
        for r in tqdm(rows, desc="Annotating"):
            jprompt = build_judge_prompt(r.task_type, r.prompt, r.response)
            jtext = judge.generate(jprompt, temperature=0.0)

            parsed = safe_parse_json(jtext)
            if parsed is None:
                # fallback label for debugging
                parsed = {"utility_label": "U0", "rationale": "Judge output invalid JSON; defaulted to U0."}

            label = normalize_label(parsed.get("utility_label", "U0"))
            rationale = (parsed.get("rationale") or "").strip()

            ex = LabeledExample(
                task_type=r.task_type,
                prompt_id=r.prompt_id,
                prompt=r.prompt,
                model_name=r.model_name,
                response=r.response,
                utility_label=label,
                rationale=rationale,
                judge_model=judge_model_name,
                meta=r.meta
            )
            out.write(ex.model_dump_json(indent=2) + "\n\n")

    print(f"Wrote: {out_path}")

if __name__ == "__main__":
    main()
