import argparse
import os
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preference_generation.noop import (  # noqa: E402
    filter_comments,
    filter_gutted_edit,
    filter_no_definitions,
    language_from_prompt,
)

DATA_PATH = os.path.join("data", "preferences")

parser = argparse.ArgumentParser(description="Filter DPO .jsonl file for no-op chosen candidates")
parser.add_argument(
    "--file",
    type=str,
    default="dpo_dataset.jsonl",
    help="Target dataset file in the preferences folder"
)
args = parser.parse_args()

if __name__ == "__main__":
    filepath = os.path.join(DATA_PATH, args.file)
    out_filepath = filepath.replace(".jsonl", "_noopfilt.jsonl")

    removed_comments = 0
    removed_definitions = 0
    removed_gutted = 0
    unknown_language = 0
    total_processed = 0
    total_kept = 0

    with open(filepath, 'r', encoding='utf-8') as f, \
         open(out_filepath, 'w', encoding='utf-8') as out_f:

        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            code = data.get("chosen", "")
            prompt = data.get("prompt", "")
            lang = data.get("language") or language_from_prompt(prompt)
            if lang is None:
                unknown_language += 1
                lang = "python"

            total_processed += 1

            fail_comments = filter_comments(code, lang)
            fail_defs = filter_no_definitions(code) if lang == "python" else False
            fail_gutted = filter_gutted_edit(code, prompt, lang) if prompt and "```" in prompt else False

            if fail_comments:
                removed_comments += 1
            if fail_defs:
                removed_definitions += 1
            if fail_gutted:
                removed_gutted += 1

            if not fail_comments and not fail_defs and not fail_gutted:
                out_f.write(json.dumps(data, ensure_ascii=False) + "\n")
                total_kept += 1

    print(f"--- Filtering Report for {args.file} ---")
    print(f"Total pairs processed: {total_processed}")
    print(f"Removed by 'all_comments': {removed_comments}")
    print(f"Removed by 'no_definitions': {removed_definitions}")
    print(f"Removed by 'gutted_edit': {removed_gutted}")
    print("(a pair can fail more than one check)")
    print(f"Total pairs kept: {total_kept}")
    if unknown_language:
        print(f"WARNING: {unknown_language} prompts with no recognizable language, checked as Python")
    print(f"Saved to: {out_filepath}")