"""Annotation utilities for labeling hallucinations."""
import json
from tqdm import tqdm
from .schema import ModelOutput, LabeledExample
from .prompts import build_judge_prompt
from .llm_clients import NIMClient


def safe_parse_json(s: str):
    # best-effort: find first {...} block
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(s[start:end+1])
    except Exception:
        return None

def main():
    in_path = "data/raw_outputs.jsonl"
    out_path = "data/labeled.jsonl"

    judge_model_name = "nim-judge"
    judge = NIMClient(
        model="meta/llama-3.1-70b-instruct",
        temperature=0.0,      # CRITICAL
        max_tokens=256,
    )

    with open(in_path, "r", encoding="utf-8") as f:
        rows = [ModelOutput.model_validate_json(line) for line in f]

    with open(out_path, "w", encoding="utf-8") as out:
        for r in tqdm(rows, desc="Annotating"):
            jprompt = build_judge_prompt(r.task_type, r.prompt, r.response)
            jtext = judge.generate(jprompt, temperature=0.0)

            parsed = safe_parse_json(jtext)
            if parsed is None:
                # fallback label for debugging
                parsed = {"utility_label": "U0", "rationale": "Judge parsing failed; defaulted to U0."}

            ex = LabeledExample(
                task_type=r.task_type,
                prompt_id=r.prompt_id,
                prompt=r.prompt,
                model_name=r.model_name,
                response=r.response,
                utility_label=parsed["utility_label"].replace("U-", "U-"),
                rationale=parsed["rationale"],
                judge_model=judge_model_name,
                meta=r.meta
            )
            out.write(ex.model_dump_json() + "\n")

    print(f"Wrote: {out_path}")

if __name__ == "__main__":
    main()
