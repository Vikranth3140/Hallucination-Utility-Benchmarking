"""Annotation utilities for labeling hallucinations."""
import json
import re
import argparse
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
from .schema import ModelOutput, LabeledExample
from .prompts import build_judge_unsupported_prompt, build_judge_utility_prompt
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

def retry_judge(judge_client, prompt, retries=2):
    """Judge with strict parsing and retry mechanism."""
    for attempt in range(retries):
        raw_output = judge_client.generate(prompt, temperature=0.0)
        parsed = safe_parse_json(raw_output)
        if parsed is not None:
            return parsed, raw_output
        
        # Retry with repair prompt if first try fails
        repair_prompt = f"{prompt}\n\n[SYSTEM]: Your previous response was not valid JSON. Please return STRICT JSON only. Format it correctly."
        prompt = repair_prompt

    return None, raw_output

def judge_unsupported(judge_client, task_type, prompt, response):
    jprompt, phash = build_judge_unsupported_prompt(task_type, prompt, response)
    parsed, raw = retry_judge(judge_client, jprompt)
    if not parsed:
        return {"unsupported_label": 0, "rationale": "Parser failed."}, raw, phash
    
    # normalize to 0 or 1
    val = parsed.get("unsupported_label", 0)
    if str(val).lower() in ("yes", "1", "true"):
        val = 1
    elif str(val).lower() in ("no", "0", "false"):
        val = 0
    else:
        val = 0 # default fallback
        
    parsed["unsupported_label"] = val
    return parsed, raw, phash

def judge_utility(judge_client, task_type, prompt, response):
    jprompt, phash = build_judge_utility_prompt(task_type, prompt, response)
    parsed, raw = retry_judge(judge_client, jprompt)
    if not parsed:
        return {"utility_label": "neutral", "rationale": "Parser failed."}, raw, phash
    
    label = parsed.get("utility_label", "").strip().lower()
    if "useful" in label or "u+" in label:
        parsed["utility_label"] = "useful"
    elif "harmful" in label or "u-" in label:
        parsed["utility_label"] = "harmful"
    else:
        parsed["utility_label"] = "neutral"
        
    return parsed, raw, phash

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/raw_outputs_PRESSURED.jsonl")
    parser.add_argument("--output", type=str, default="data/labeled_PRESSURED.jsonl")
    parser.add_argument("--log_dir", type=str, default="outputs/logs")
    args = parser.parse_args()

    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    raw_log = open(Path(args.log_dir) / "judge_raw_responses.jsonl", "w", encoding="utf-8")

    judge_model_name = "nim-judge"
    judge = NIMClient(
        model="meta/llama-3.1-70b-instruct",
        temperature=0.0,      # CRITICAL - keep deterministic
        max_tokens=256,
    )

    with open(args.input, "r", encoding="utf-8") as f:
        content = f.read()
        json_strings = [s.strip() for s in content.split('\n\n') if s.strip() and '{' in s]
        rows = []
        for js in json_strings:
            try:
                rows.append(ModelOutput.model_validate_json(js))
            except Exception as e:
                pass


    with open(args.output, "w", encoding="utf-8") as out:
        for r in tqdm(rows, desc="Annotating"):
            # Stage 1: Unsupported
            unsup_parsed, unsup_raw, unsup_phash = judge_unsupported(judge, r.task_type, r.prompt, r.response)
            unsupported_label = unsup_parsed.get("unsupported_label", 0)
            rationale = unsup_parsed.get("rationale", "")
            
            # Log
            raw_log.write(json.dumps({"prompt_id": r.prompt_id, "stage": "unsupported", "raw": unsup_raw}) + "\n")

            # Stage 2: Utility (Conditional)
            utility_label = None
            if unsupported_label == 1:
                util_parsed, util_raw, util_phash = judge_utility(judge, r.task_type, r.prompt, r.response)
                utility_label = util_parsed.get("utility_label", "neutral")
                rationale += " | Util: " + util_parsed.get("rationale", "")
                
                raw_log.write(json.dumps({"prompt_id": r.prompt_id, "stage": "utility", "raw": util_raw}) + "\n")

            try:
                ex = LabeledExample(
                    task_type=r.task_type,
                    prompt_id=r.prompt_id,
                    prompt=r.prompt,
                    model_name=r.model_name,
                    response=r.response,
                    unsupported_label=unsupported_label,
                    utility_label=utility_label,
                    rationale=rationale,
                    judge_model=judge_model_name,
                    meta=r.meta
                )
                out.write(ex.model_dump_json() + "\n")
            except Exception as e:
                print(f"Failed to create LabeledExample for {r.prompt_id}: {e}")

    raw_log.close()
    print(f"Wrote: {args.output}")

if __name__ == "__main__":
    main()
