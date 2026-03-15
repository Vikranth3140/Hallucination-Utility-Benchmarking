"""Migration script to convert old labels (U+, U0, U-) to the 2-stage schema."""
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from .schema import LabeledExample

def parse_args():
    parser = argparse.ArgumentParser(description="Migrate old labels to the new two-stage schema.")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file")
    parser.add_argument("--ambiguous-log", type=str, default="outputs/logs/ambiguous_migrations.jsonl", help="Log file for ambiguous remappings.")
    return parser.parse_args()

def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    log_path = Path(args.ambiguous_log)
    
    log_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    out_f = open(output_path, "w", encoding="utf-8")
    log_f = open(log_path, "w", encoding="utf-8")
    
    num_migrated = 0
    num_ambiguous = 0
    
    with open(input_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Migrating"):
            line = line.strip()
            if not line:
                continue
            
            data = json.loads(line)
            
            if "unsupported_label" in data:
                # Already migrated or new schema
                try:
                    ex = LabeledExample(**data)
                    out_f.write(ex.model_dump_json() + "\n")
                    num_migrated += 1
                except Exception as e:
                    print(f"Validation failed for already migrated line: {e}")
                continue
            
            old_label = data.get("utility_label")
            unsupported_label = None
            new_utility_label = None
            is_ambiguous = False
            
            if old_label == "U+":
                unsupported_label = 1
                new_utility_label = "useful"
            elif old_label == "U-":
                unsupported_label = 1
                new_utility_label = "harmful"
            elif old_label == "U0":
                # This could mean no hallucination (unsupported=0) OR neutral hallucination
                # If no hallucination flag exists, preserve as unresolved/review-needed.
                # Since the schema requires utility_label=None if unsupported=0, or 'neutral' if unsupported=1,
                # we will default to unsupported=0, but log it as ambiguous.
                unsupported_label = 0
                new_utility_label = None
                is_ambiguous = True
            else:
                # Unknown old label, default to unsupported=0
                unsupported_label = 0
                new_utility_label = None
                is_ambiguous = True
                
            data["unsupported_label"] = unsupported_label
            data["utility_label"] = new_utility_label
            
            if is_ambiguous:
                log_f.write(json.dumps(data) + "\n")
                num_ambiguous += 1
            
            # Write to output assuming we "handled" it
            try:
                ex = LabeledExample(**data)
                out_f.write(ex.model_dump_json() + "\n")
                num_migrated += 1
            except Exception as e:
                print(f"Failed to create LabeledExample: {e}\nData: {data}")
                
    out_f.close()
    log_f.close()
    
    print(f"Successfully migrated {num_migrated} examples.")
    if num_ambiguous > 0:
        print(f"Logged {num_ambiguous} ambiguous examples to {log_path}.")

if __name__ == "__main__":
    main()
