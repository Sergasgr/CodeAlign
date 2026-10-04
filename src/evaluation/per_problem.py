import os
import ast
import json
import argparse
import glob
from datasets import load_dataset
import multiprocessing

TIMEOUT_SECONDS = 3
TASK = "humaneval-chat"

def execute_code(code: str, return_dict: dict):
    with open(os.devnull, "w") as devnull:
        os.dup2(devnull.fileno(), 1)
        os.dup2(devnull.fileno(), 2)
    try:
        exec_globals = {}
        exec(code, exec_globals)
        return_dict['passed'] = True
        return_dict['error'] = None
    except BaseException as e: 
        return_dict['passed'] = False
        return_dict['error'] = e.__class__.__name__

def prompt_helpers(prompt: str, entry_point: str) -> str:
    tree = ast.parse(prompt)
    helpers = [
        n for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name != entry_point
    ]
    if not helpers:
        return ""
    imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    return "\n".join(ast.get_source_segment(prompt, n) for n in imports + helpers) + "\n"

def evaluate_single_problem(solution: str, problem: dict):
    reference = "\n" + prompt_helpers(problem["prompt"], problem["entry_point"]) + problem["test"] + "\n" + f"check({problem['entry_point']})"
    full_code = solution + "\n" + reference

    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    p = multiprocessing.Process(target=execute_code, args=(full_code, return_dict))
    p.start()
    p.join(TIMEOUT_SECONDS)

    if p.is_alive():
        p.kill()
        p.join()
        manager.shutdown()
        return False, "TimeoutError"

    result = return_dict.get('passed', False), return_dict.get('error', "UnknownError")
    manager.shutdown()
    return result

def harness_pass_at_1(run_name: str, language: str) -> float | None:
    path = os.path.join("data", "evaluation", f"{run_name}_metrics_{language}.json")
    try:
        with open(path, encoding="utf-8") as f:
            metrics = json.load(f)
    except FileNotFoundError:
        return None
    return metrics.get(metrics.get("config", {}).get("tasks", TASK), {}).get("pass@1")

def main():
    parser = argparse.ArgumentParser(description="Re-execute saved generations against HumanEval tests.")
    parser.add_argument("--language", type=str, default="python", help="Language to evaluate.")
    args = parser.parse_args()
    if args.language != "python":
        raise SystemExit("Only 'python' (HumanEval) is supported in v1.1.")

    dataset = load_dataset("openai/openai_humaneval", split="test")
    search_pattern = os.path.join("data", "evaluation", f"*_generations_{args.language}.json")
    generation_files = sorted(glob.glob(search_pattern))
    mismatches = []

    for filepath in generation_files:
        run_name = os.path.basename(filepath).replace(f"_generations_{args.language}.json", "")
        print(f"\nEvaluating run: {run_name}")

        with open(filepath, 'r', encoding='utf-8') as f:
            generations = json.load(f)
        if len(generations) != len(dataset):
            print(f"Skipping: {len(generations)} generations for {len(dataset)} problems (a --limit run?)")
            continue

        results_per_problem = {}
        total_passed = 0

        for task_idx, problem in enumerate(dataset):
            task_id = problem["task_id"]

            solution_data = generations[task_idx] if isinstance(generations, list) else generations.get(task_id, [])
            solution = solution_data[0] if isinstance(solution_data, list) else solution_data

            passed, error = evaluate_single_problem(solution, problem)

            results_per_problem[task_id] = {
                "passed": passed,
                "error_class": error,
                "entry_point": problem["entry_point"],
            }
            if passed:
                total_passed += 1

        pass_at_1 = (total_passed / len(dataset)) * 100

        output_file = os.path.join("data", "evaluation", f"{run_name}_per_problem.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_per_problem, f, indent=4)

        print(f"Pass@1: {pass_at_1:.2f}% ({total_passed}/{len(dataset)})")
        harness = harness_pass_at_1(run_name, args.language)
        if harness is None:
            print("No metrics file to check against")
        elif round(harness * len(dataset)) == total_passed:
            print(f"✓ Matches the harness ({100 * harness:.2f}%)")
        else:
            print(f"⚠ MISMATCH: harness {100 * harness:.2f}% ({round(harness * len(dataset))}/{len(dataset)})")
            mismatches.append(run_name)
        print(f"Detailed results saved to {output_file}")

    if mismatches:
        print(f"\n⚠ Runs that do not reproduce the harness pass@1: {mismatches}. Usually a solution near the "
              f"{TIMEOUT_SECONDS}s limit; notebook 06 flags these runs.")

if __name__ == "__main__":
    main()
