import argparse
import json
import os
from pathlib import Path
import wandb

from dotenv import load_dotenv
from tqdm import tqdm

from src.data_curation.curation_config import DS_PATH
from src.preference_generation.preference_generation_config import (
    DPO_DS,
    DPO_REPORT,
    WANDB_RUN_NAME,
    W_EXEC,
    W_COMPLEXITY,
    W_LINT
)
from src.preference_generation.preference_orchestrator import PreferenceOrchestrator

load_dotenv()

def main():
    wandb_api_key = os.getenv("WANDB_API_KEY")
    wandb_project = os.getenv("WANDB_PROJECT")
    wandb_entity = os.getenv("WANDB_ENTITY")

    if wandb_api_key:
        wandb.login(key=wandb_api_key)
    else:
        raise ValueError("WANDB_API_KEY not found as environment variable")
    
    parser = argparse.ArgumentParser(description="Generate DPO Preferences")
    parser.add_argument(
        "--reward_mode", 
        type=str, 
        choices=["composite", "execution_only"], 
        default="composite",
        help="Define los pesos del reward para generar el dataset."
    )
    args = parser.parse_args()
    
    if args.reward_mode == "execution_only":
        w_exec, w_comp, w_lint = 1.0, 0.0, 0.0
        dpo_output_path = DPO_DS.replace(".jsonl", "_exec_only.jsonl")
        report_output_path = DPO_REPORT.replace(".jsonl", "_exec_only.jsonl")
        run_name = f"{WANDB_RUN_NAME}-exec-only"
        tags = ["preference-generation", "sandbox", "execution-only"]
    else:
        w_exec, w_comp, w_lint = W_EXEC, W_COMPLEXITY, W_LINT
        dpo_output_path = DPO_DS
        report_output_path = DPO_REPORT
        run_name = f"{WANDB_RUN_NAME}-composite"
        tags = ["preference-generation", "sandbox", "composite-reward"]

    wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        name=run_name,
        tags=tags,
    )
    
    print(f"Initializing orchestrator in mode: {args.reward_mode}")
    print(f"Weights -> Exec: {w_exec}, Comp: {w_comp}, Lint: {w_lint}")

    all_data = []
    with open(DS_PATH, "r", encoding="utf-8") as ds:
        for line in ds:
            all_data.append(json.loads(line))

    print(f"Loaded {len(all_data)} prompts from {DS_PATH}")

    print("Initializing SFT model and Docker sandbox...")
    orchestrator = PreferenceOrchestrator(
        w_exec=w_exec, 
        w_complexity=w_comp, 
        w_lint=w_lint
    )

    stats = {
        "total_prompts": len(all_data),
        "case_a": 0,
        "case_b": 0,
        "case_c": 0,
        "case_c_tied": 0,
        "pairs_generated": 0,
        "pairs_discarded": 0,
    }
    score_accum = {
        "chosen_score": 0.0,
        "rejected_score": 0.0,
        "chosen_cc": 0.0,
        "rejected_cc": 0.0,
        "chosen_lint": 0.0,
        "rejected_lint": 0.0,
    }

    Path(DPO_DS).parent.mkdir(parents=True, exist_ok=True)

    with open(DPO_DS, "w", encoding="utf-8") as dpo_f, \
         open(DPO_REPORT, "w", encoding="utf-8") as report_f:

        for data in tqdm(all_data, desc="Generating DPO pairs"):
            prompt = data["messages"][0]["content"]
            language = data["metadata"]["language"]

            result = orchestrator.create_preference_pair(prompt, language)

            report_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            report_f.flush()
            
            case = result.get("case", "B")
            case_key = f"case_{case.lower()}"
            stats[case_key] = stats.get(case_key, 0) + 1

            if result.get("discarded"):
                stats["pairs_discarded"] += 1
                continue

            dpo_pair = {
                "prompt": result["prompt"],
                "chosen": result["chosen"],
                "rejected": result["rejected"],
            }
            dpo_f.write(json.dumps(dpo_pair, ensure_ascii=False) + "\n")
            dpo_f.flush()

            stats["pairs_generated"] += 1
            score_accum["chosen_score"] += result.get("chosen_score", 0)
            score_accum["rejected_score"] += result.get("rejected_score", 0)
            score_accum["chosen_cc"] += float(result.get("chosen_complexity", 0) or 0)
            score_accum["rejected_cc"] += float(result.get("rejected_complexity", 0) or 0)
            score_accum["chosen_lint"] += float(result.get("chosen_lint_errors", 0) or 0)
            score_accum["rejected_lint"] += float(result.get("rejected_lint_errors", 0) or 0)

            if stats["pairs_generated"] % 500 == 0:
                n_so_far = max(stats["pairs_generated"], 1)
                wandb.log({
                    "pairs_generated": stats["pairs_generated"],
                    "pairs_discarded": stats["pairs_discarded"],
                    "case_a": stats["case_a"],
                    "case_b": stats["case_b"],
                    "case_c": stats["case_c"],
                    "avg_chosen_score": round(score_accum["chosen_score"] / n_so_far, 4),
                    "avg_rejected_score": round(score_accum["rejected_score"] / n_so_far, 4),
                })

    n = max(stats["pairs_generated"], 1)
    avg_stats = {
        "avg_chosen_score": round(score_accum["chosen_score"] / n, 4),
        "avg_rejected_score": round(score_accum["rejected_score"] / n, 4),
        "avg_chosen_cc": round(score_accum["chosen_cc"] / n, 4),
        "avg_rejected_cc": round(score_accum["rejected_cc"] / n, 4),
        "avg_chosen_lint_errors": round(score_accum["chosen_lint"] / n, 4),
        "avg_rejected_lint_errors": round(score_accum["rejected_lint"] / n, 4),
    }

    all_stats = {**stats, **avg_stats}

    wandb.log(all_stats)
    wandb.finish()

    print(f"\n{'=' * 60}")
    print("Phase 3 — Preference Generation Complete")
    print(f"{'=' * 60}")
    print(f"Total prompts processed: {stats['total_prompts']}")
    print(f"Pairs generated:        {stats['pairs_generated']}")
    print(f"Pairs discarded:        {stats['pairs_discarded']}")
    print(f"\nCase breakdown:")
    print(f"  Case A (one pass, one fail): {stats['case_a']}")
    print(f"  Case B (both fail):          {stats['case_b']}")
    print(f"  Case C (both pass, ranked):  {stats['case_c']}")
    if stats.get("case_c_tied"):
        print(f"  Case C tied (discarded):     {stats['case_c_tied']}")
    print(f"\nAvg composite score — chosen: {avg_stats['avg_chosen_score']:.4f}"
          f" | rejected: {avg_stats['avg_rejected_score']:.4f}")
    print(f"Avg cyclomatic complexity — chosen: {avg_stats['avg_chosen_cc']:.2f}"
          f" | rejected: {avg_stats['avg_rejected_cc']:.2f}")
    print(f"Avg lint errors — chosen: {avg_stats['avg_chosen_lint_errors']:.2f}"
          f" | rejected: {avg_stats['avg_rejected_lint_errors']:.2f}")
    print(f"\nDPO dataset: {DPO_DS}")
    print(f"Full report: {DPO_REPORT}")

if __name__ == "__main__":
    main()