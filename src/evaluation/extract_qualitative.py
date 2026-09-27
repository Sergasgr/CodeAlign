import argparse
import json
from pathlib import Path

from src.data_curation.validators import linter_check, validate_syntax
from src.evaluation.evaluation_config import (
    DPO_ABLATION_GENERATIONS,
    DPO_COMPOSITE_GENERATIONS,
    QUALITATIVE_SAMPLES_MD,
    TARGET_SAMPLES,
    COMPLEXITY_GAP,
)

def has_syntax_error(code: str, lang: str) -> bool:
    is_valid, error = validate_syntax(code, lang)
    return not is_valid and error == "Syntax error detected by tree-sitter"

def extract_qualitative_samples(lang: str) -> None:
    ablation_file = DPO_ABLATION_GENERATIONS.format(lang=lang)
    composite_file = DPO_COMPOSITE_GENERATIONS.format(lang=lang)
    output_md = QUALITATIVE_SAMPLES_MD.format(lang=lang)
    
    try:
        with open(ablation_file, "r", encoding="utf-8") as f:
            ablation_gens = json.load(f)

        with open(composite_file, "r", encoding="utf-8") as f:
            composite_gens = json.load(f)
    except FileNotFoundError as e:
        print(f"Error loading generation files: {e}")
        return

    Path(output_md).parent.mkdir(parents=True, exist_ok=True)
    found = 0

    with open(output_md, "w", encoding="utf-8") as out:
        out.write(f"# Qualitative Samples ({lang}): Ablation DPO vs. Composite Reward DPO\n\n")
        out.write("Problems where the composite-reward model's code has a cyclomatic complexity at least\n")
        out.write(f"{COMPLEXITY_GAP} lower than the execution-only model's; both are syntactically valid.\n")
        out.write("Selected by complexity gap for illustration. Correctness is not checked here (see pass@1).\n\n")
        out.write("---\n\n")

        for idx, (abl_samples, comp_samples) in enumerate(zip(ablation_gens, composite_gens)):
            code_abl = abl_samples[0]
            code_comp = comp_samples[0]
            if has_syntax_error(code_abl, lang) or has_syntax_error(code_comp, lang):
                continue

            metrics_abl = linter_check(code_abl, lang)
            metrics_comp = linter_check(code_comp, lang)

            cc_abl = metrics_abl.get("complexity")
            cc_comp = metrics_comp.get("complexity")
            lint_abl = metrics_abl.get("lint_errors")
            lint_comp = metrics_comp.get("lint_errors")
            
            if None in (cc_abl, cc_comp, lint_abl, lint_comp):
                continue

            if cc_abl - cc_comp >= COMPLEXITY_GAP: # type: ignore
                found += 1
                problem = f"HumanEval/{idx}" if lang == "python" else f"MultiPL-E {lang} #{idx}"
                out.write(f"### {problem}\n\n")
                out.write(f"| Metric | Ablation (exec-only) | Composite |\n")
                out.write(f"|--------|---------------------|-----------|\n")
                out.write(f"| Cyclomatic complexity | {cc_abl} | {cc_comp} |\n")
                out.write(f"| Lint errors | {lint_abl} | {lint_comp} |\n\n")
                out.write(f"**DPO Ablation (execution-only reward):**\n")
                out.write(f"```{lang}\n{code_abl.strip()}\n```\n\n")
                out.write(f"**DPO Composite Reward:**\n")
                out.write(f"```{lang}\n{code_comp.strip()}\n```\n\n")
                out.write("---\n\n")

                if found >= TARGET_SAMPLES:
                    break

    print(f"Extracted {found} qualitative samples for {lang} → {output_md}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract qualitative samples for a specific language.")
    parser.add_argument(
        "--language", 
        type=str, 
        required=True, 
        help="Language of the evaluated generations (e.g., python, rust, go)"
    )
    args = parser.parse_args()
    
    extract_qualitative_samples(args.language)
