"""Generate LLM outputs for benchmarking."""
import json
import os
from tqdm import tqdm
from .schema import ModelOutput
from .prompts import load_prompts, build_generation_prompt
from .llm_clients import NIMClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    prompts = load_prompts("data/prompts.json")

    # Add as many model clients as you want here:
    models = [
        ("nim-llama", NIMClient(
            model="meta/llama-3.1-8b-instruct",
            temperature=0.7,
            max_tokens=512,
        )),
        ("nim-qwen", NIMClient(
            model="qwen/qwen2.5-7b-instruct",
            temperature=0.7,
            max_tokens=512,
        )),
    ]
    out_path = "data/raw_outputs.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for p in tqdm(prompts, desc="Generating"):
            gen_prompt = build_generation_prompt(p.task_type, p.prompt)
            for model_name, client in models:
                text = client.generate(gen_prompt, temperature=0.7)
                # Extract RESPONSE: block if present (simple)
                resp = text
                if "RESPONSE:" in text:
                    resp = text.split("RESPONSE:", 1)[1].strip()

                item = ModelOutput(
                    task_type=p.task_type,
                    prompt_id=p.prompt_id,
                    prompt=p.prompt,
                    model_name=model_name,
                    response=resp,
                    meta={"temperature": 0.7}
                )
                f.write(item.model_dump_json() + "\n")

    print(f"Wrote: {out_path}")

if __name__ == "__main__":
    main()
