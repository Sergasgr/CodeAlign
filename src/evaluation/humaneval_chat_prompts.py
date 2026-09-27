import argparse
import json
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
INSTRUCTION = "Please provide a self-contained Python script that solves the following problem in a markdown code block:"
RESPONSE = "Below is a Python script with a self-contained function that solves the problem and passes corresponding tests:"
_SPLITTER = "-[[]]-this-is-really-our-highest-priority-[[]]-" 

def render(tokenizer, problem: str) -> str:
    user = f"{INSTRUCTION}\n```\n{problem.strip()}\n```\n"
    assistant = f"{RESPONSE}\n```python\n{_SPLITTER}\n```\n"
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}],
        tokenize=False,
    )
    return text.split(_SPLITTER)[0]

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=MODEL, help="Tokenizer whose chat template renders the prompts")
    parser.add_argument("--out", required=True, help="JSON file {task_id: prompt}")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    problems = load_dataset("openai/openai_humaneval", split="test")
    prompts = {row["task_id"]: render(tokenizer, row["prompt"]) for row in problems}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(prompts, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{len(prompts)} chat prompts ({args.model} template) → {args.out}")

if __name__ == "__main__":
    main()
