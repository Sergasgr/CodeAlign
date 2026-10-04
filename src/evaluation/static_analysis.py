import argparse
import json
import glob
import os
from pathlib import Path

from src.data_curation.validators import linter_check, validate_syntax
from src.evaluation.evaluation_config import (
    BASE_GENERATIONS,
    SFT_GENERATIONS,
    DPO_COMPOSITE_GENERATIONS,
    DPO_ABLATION_GENERATIONS,
    DPO_COMPOSITE_MATCHED_GENERATIONS,
    EVAL_DIR,
    STATIC_ANALYSIS_RESULTS,
)

def get_models_dict(lang: str) -> dict[str, str]:
    return {
        "base": BASE_GENERATIONS.format(lang=lang),
        "sft": SFT_GENERATIONS.format(lang=lang),
        "dpo_composite": DPO_COMPOSITE_GENERATIONS.format(lang=lang),
        "dpo_ablation": DPO_ABLATION_GENERATIONS.format(lang=lang),
        "dpo_composite_matched": DPO_COMPOSITE_MATCHED_GENERATIONS.format(lang=lang),
    }

def analyze_generations(models: dict[str, str], output_path: str, lang: str, keep_other_models: bool = False) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    total = 0

    kept_lines = []
    if keep_other_models and os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            kept_lines = [line for line in f if line.strip() and json.loads(line).get("model") not in models]

    with open(output_path, "w", encoding="utf-8") as out:
        out.writelines(kept_lines)
        for model_name, gen_path in models.items():
            try:
                with open(gen_path, "r", encoding="utf-8") as f:
                    generations = json.load(f)
            except FileNotFoundError:
                print(f"Skipping {model_name}: {gen_path} not found")
                continue

            for problem_idx, samples in enumerate(generations):
                for sample_idx, code_str in enumerate(samples):
                    is_valid, error = validate_syntax(code_str, lang)
                    if not is_valid and error == "Syntax error detected by tree-sitter":
                        result = {"complexity": None, "lint_errors": None}
                    else:
                        result = linter_check(code_str, lang)
                    record = {
                        "model": model_name,
                        "problem_idx": problem_idx,
                        "sample_idx": sample_idx,
                        "complexity": result.get("complexity"),
                        "lint_errors": result.get("lint_errors"),
                    }
                    out.write(json.dumps(record) + "\n")
                    total += 1

    print(f"{total} samples analyzed for {lang} → {output_path}"
          + (f" ({len(kept_lines)} rows of other models kept)" if keep_other_models else ""))

def load_all_generations(base_dir=EVAL_DIR, language="python") -> dict[str, str]:
    search_pattern = os.path.join(base_dir, f"*_generations_{language}.json")
    generation_files = glob.glob(search_pattern)

    found = {}
    for filepath in generation_files:
        filename = os.path.basename(filepath)
        name = filename.replace(f"_generations_{language}.json", "")
        found[name] = filepath

    v1_order = list(get_models_dict(language))
    return {name: found[name] for name in sorted(found, key=lambda n: (n not in v1_order, v1_order.index(n) if n in v1_order else 0, n))}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run static analysis on evaluation generations.")
    parser.add_argument(
        "--language",
        type=str,
        required=True,
        help="Language of the evaluated generations (e.g., python, rust, go)"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Only (re)analyze these models (<name> of <name>_generations_<lang>.json) and keep every "
             "other model's rows in the results file. Default: all generation files, file rewritten.",
    )
    args = parser.parse_args()

    models = load_all_generations(language=args.language)
    if args.models:
        missing = [m for m in args.models if m not in models]
        if missing:
            raise SystemExit(f"No generations file for {missing} in {EVAL_DIR}")
        models = {m: models[m] for m in args.models}
    output_path = STATIC_ANALYSIS_RESULTS.format(lang=args.language)

    analyze_generations(models, output_path, args.language, keep_other_models=bool(args.models))
