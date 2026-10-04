import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preference_generation.noop import filter_comments, filter_no_definitions, filter_gutted_edit  # noqa: E402

DATA_PATH = os.path.join("data", "preferences")
DEFAULT_REPORTS = [os.path.join(DATA_PATH, "dpo_report.jsonl"), os.path.join(DATA_PATH, "dpo_report_exec_only.jsonl")]
DEFAULT_OUT = os.path.join(DATA_PATH, "noop_audit.json")
FLAGS = ["all_comments", "no_definitions", "gutted_edit", "any_flag"]
CASE_ORDER = ["A", "B", "C"]

results = {}
"""
Dictionary with the results:
    Key: Filename (e.g., "dpo_report.jsonl")
        Key: Case type (A, B, C)
            Key: Programming Language (e.g., "python", "java")
            Values:
                chosen_total: Total number of chosen samples processed
                rejected_total: Total number of rejected samples processed
                edit_total: Pairs whose prompt is an edit (the only ones gutted_edit can flag)
                chosen_flags: Dictionary with counts for each flag
                    all_comments: Count of chosen samples failing the comments filter
                    no_definitions: Count of chosen samples failing the definitions filter
                    gutted_edit: Count of chosen samples failing the gutted edit filter
                    any_flag: Count of chosen samples failing at least one of the three
                rejected_flags: Dictionary with counts for each flag
                    (same keys, for the rejected samples)
                any_flag_discordant: pairs where only the chosen (chosen_only) or only the
                    rejected (rejected_only) candidate has a flag
"""

def candidate_flags(code: str, prompt: str, lang: str) -> dict[str, bool]:
    flags = {
        "all_comments": filter_comments(code, lang),
        "no_definitions": lang == "python" and filter_no_definitions(code),
        "gutted_edit": bool(prompt) and "```" in prompt and filter_gutted_edit(code, prompt, lang),
    }
    flags["any_flag"] = any(flags.values())
    return flags

def audit(filepaths: list[str]) -> dict:
    for filepath in filepaths:
        filename = os.path.basename(filepath)
        results[filename] = {}
        skipped = 0

        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                if data.get("discarded"):
                    skipped += 1
                    continue
                case = data.get("case")
                lang = data.get("language")

                if case not in results[filename]:
                    results[filename][case] = {}

                if lang not in results[filename][case]:
                    results[filename][case][lang] = {
                        "chosen_total": 0,
                        "rejected_total": 0,
                        "edit_total": 0,
                        "chosen_flags": {flag: 0 for flag in FLAGS},
                        "rejected_flags": {flag: 0 for flag in FLAGS},
                        "any_flag_discordant": {"chosen_only": 0, "rejected_only": 0},
                    }

                metrics = results[filename][case][lang]

                chosen_code = data.get("chosen") or ""
                rejected_code = data.get("rejected") or ""
                prompt = data.get("prompt") or ""

                metrics["chosen_total"] += 1
                metrics["rejected_total"] += 1
                metrics["edit_total"] += prompt.startswith("Here is the current file:")

                flags = {side: candidate_flags(code, prompt, lang)
                         for side, code in (("chosen", chosen_code), ("rejected", rejected_code))}
                for side, side_flags in flags.items():
                    for flag, hit in side_flags.items():
                        metrics[f"{side}_flags"][flag] += hit
                chosen_any, rejected_any = flags["chosen"]["any_flag"], flags["rejected"]["any_flag"]
                metrics["any_flag_discordant"]["chosen_only"] += chosen_any and not rejected_any
                metrics["any_flag_discordant"]["rejected_only"] += rejected_any and not chosen_any

        print(f"{filename}: {skipped:,} discarded prompts skipped (no pair)")
    return results

def print_report(results: dict) -> None:
    for filename, cases in results.items():
        print(f"\n{'='*50}\nFile: {filename}\n{'='*50}")
        for case in sorted(cases, key=lambda c: CASE_ORDER.index(c) if c in CASE_ORDER else len(CASE_ORDER)):
            for lang, metrics in sorted(cases[case].items()):
                chosen_tot = metrics["chosen_total"]
                rejected_tot = metrics["rejected_total"]

                print(f"\nCase: {case} | Language: {lang}")
                print(f"Total processed -> Chosen: {chosen_tot}, Rejected: {rejected_tot} (edit prompts: {metrics['edit_total']})")

                if chosen_tot > 0 and rejected_tot > 0:
                    print(f"{'Flag':<20} | {'% in Chosen':<15} | {'% in Rejected'}")
                    print("-" * 55)
                    for flag in FLAGS:
                        pct_chosen = (metrics["chosen_flags"][flag] / chosen_tot) * 100
                        pct_rejected = (metrics["rejected_flags"][flag] / rejected_tot) * 100

                        print(f"{flag:<20} | {pct_chosen:>6.1f}%{'':<8} | {pct_rejected:>6.1f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit preference pairs for no-op chosen/rejected candidates.")
    parser.add_argument("reports", nargs="*", default=DEFAULT_REPORTS, help="dpo_report*.jsonl files")
    parser.add_argument("--out", default=DEFAULT_OUT, help="JSON written for notebook 06")
    args = parser.parse_args()

    missing = [p for p in args.reports if not os.path.isfile(p)]
    if missing:
        sys.exit(f"Report file(s) not found: {missing}. Run from the repo root.")

    print_report(audit(args.reports))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.out}")