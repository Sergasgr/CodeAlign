import json
from src.evaluation.evaluation_config import (
    BASE_EVALUATED_MODEL,
    SFT_EVALUATED_MODEL,
    DPO_ABLATION_EVALUATED_MODEL,
    DPO_COMPOSITE_REWARD_EVALUATED_MODEL,
    RESULTS_EVALUATION
)
from src.data_curation.validators import linter_check

MODELS = {
    "BASE": BASE_EVALUATED_MODEL,
    "SFT": SFT_EVALUATED_MODEL,
    "DPO_ABLATION": DPO_ABLATION_EVALUATED_MODEL,
    "DPO_REWARD": DPO_COMPOSITE_REWARD_EVALUATED_MODEL,
}

def evaluation_result(model: str, problem_id: str, cyclomatic_complexity: float, linter_errors: int):
    return {
        "model": model,
        "problem_id": problem_id,
        "complexity": cyclomatic_complexity,
        "linter_errors": linter_errors
    }

with open(RESULTS_EVALUATION, 'w', encoding='utf-8') as r:
    for name, path in MODELS.items():
        try:
            with open(path, 'r', encoding='utf-8') as f: 
                generations = json.load(f)
                for problem_idx, samples in enumerate(generations):
                    for sample_idx, code_str in enumerate(samples):
                        result = linter_check(code_str, "python") #deberia haber soporte multilenguaje hay que cambiar
                        unique_id = f"{name}_prob{problem_idx}_samp{sample_idx}"
                        r.write(json.dumps(evaluation_result(
                            name, unique_id, result["complexity"], result["lint_errors"]
                        )) + "\n")
        except FileNotFoundError:
            print(f"Warning: Generations file not found for {name} in {path}")

print(f"Static metrics calculated and saved in {RESULTS_EVALUATION}")   

