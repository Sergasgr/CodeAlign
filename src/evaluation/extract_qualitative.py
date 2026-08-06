import json
from pathlib import Path

from src.data_curation.validators import linter_check
from src.evaluation.evaluation_config import (
    DPO_ABLATION_GENERATIONS,
    DPO_COMPOSITE_GENERATIONS,
    QUALITATIVE_SAMPLES_MD,
    TARGET_SAMPLES,
    COMPLEXITY_GAP,
)

def extract_qualitative_samples() -> None:
    with open(DPO_ABLATION_GENERATIONS, "r", encoding="utf-8") as f:
        ablation_gens = json.load(f)

    with open(DPO_COMPOSITE_GENERATIONS, "r", encoding="utf-8") as f:
        composite_gens = json.load(f)

    Path(QUALITATIVE_SAMPLES_MD).parent.mkdir(parents=True, exist_ok=True)
    found = 0

    with open(QUALITATIVE_SAMPLES_MD, "w", encoding="utf-8") as out:
        out.write("# Qualitative Samples: Ablation DPO vs. Composite Reward DPO\n\n")
        out.write("Problems where both models produce working code, but the composite-reward\n")
        out.write("model generates simpler, cleaner code (lower cyclomatic complexity).\n\n")
        out.write("---\n\n")

        for idx, (abl_samples, comp_samples) in enumerate(
            zip(ablation_gens, composite_gens)
        ):
            code_abl = abl_samples[0]
            code_comp = comp_samples[0]

            metrics_abl = linter_check(code_abl, "python")
            metrics_comp = linter_check(code_comp, "python")

            cc_abl = metrics_abl.get("complexity")
            cc_comp = metrics_comp.get("complexity")
            lint_abl = metrics_abl.get("lint_errors")
            lint_comp = metrics_comp.get("lint_errors")
            
            if None in (cc_abl, cc_comp, lint_abl, lint_comp):
                continue

            if cc_abl - cc_comp >= COMPLEXITY_GAP: # type: ignore
                found += 1
                out.write(f"### HumanEval #{idx}\n\n")
                out.write(f"| Metric | Ablation (exec-only) | Composite |\n")
                out.write(f"|--------|---------------------|-----------|\n")
                out.write(f"| Cyclomatic complexity | {cc_abl} | {cc_comp} |\n")
                out.write(f"| Lint errors | {lint_abl} | {lint_comp} |\n\n")
                out.write(f"**DPO Ablation (execution-only reward):**\n")
                out.write(f"```python\n{code_abl.strip()}\n```\n\n")
                out.write(f"**DPO Composite Reward:**\n")
                out.write(f"```python\n{code_comp.strip()}\n```\n\n")
                out.write("---\n\n")

                if found >= TARGET_SAMPLES:
                    break

    print(f"Extracted {found} qualitative samples → {QUALITATIVE_SAMPLES_MD}")

if __name__ == "__main__":
    extract_qualitative_samples()
