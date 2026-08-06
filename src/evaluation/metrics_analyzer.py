import json
from pathlib import Path

from src.data_curation.validators import linter_check
from src.evaluation.evaluation_config import (
    BASE_GENERATIONS,
    SFT_GENERATIONS,
    DPO_COMPOSITE_GENERATIONS,
    DPO_ABLATION_GENERATIONS,
    STATIC_ANALYSIS_RESULTS,
)

MODELS = {
    "base": BASE_GENERATIONS,
    "sft": SFT_GENERATIONS,
    "dpo_composite": DPO_COMPOSITE_GENERATIONS,
    "dpo_ablation": DPO_ABLATION_GENERATIONS,
}

def analyze_generations(models: dict[str, str], output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    total = 0

    with open(output_path, "w", encoding="utf-8") as out:
        for model_name, gen_path in models.items():
            try:
                with open(gen_path, "r", encoding="utf-8") as f:
                    generations = json.load(f)
            except FileNotFoundError:
                print(f"Skipping {model_name}: {gen_path} not found")
                continue

            for problem_idx, samples in enumerate(generations):
                for sample_idx, code_str in enumerate(samples):
                    result = linter_check(code_str, "python")
                    record = {
                        "model": model_name,
                        "problem_idx": problem_idx,
                        "sample_idx": sample_idx,
                        "complexity": result.get("complexity"),
                        "lint_errors": result.get("lint_errors"),
                    }
                    out.write(json.dumps(record) + "\n")
                    total += 1

    print(f"{total} samples analyzed → {output_path}")

if __name__ == "__main__":
    analyze_generations(MODELS, STATIC_ANALYSIS_RESULTS)
