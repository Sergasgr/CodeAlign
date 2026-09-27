"""Build the size-matched control arm for the DPO reward ablation.

The execution-only run trains on far fewer pairs than the composite run (it can only use
Case A), so "composite vs. execution-only" mixes the reward with data volume. This script
draws a random subset of the composite pairs with exactly the execution-only pair count.
Training DPO on it with the same seed / epochs / hyperparameters isolates the reward:

    composite (size-matched) vs. execution-only  -> effect of the reward at a fixed data budget
    composite (full) vs. composite (size-matched) -> effect of the extra pairs the reward rescues

Usage:
    uv run python scripts/make_size_matched_pairs.py \
        --composite data/preferences/dpo_dataset.jsonl \
        --match     data/preferences/<execution-only pairs>.jsonl
"""

import argparse
import random
from pathlib import Path


def read_nonempty_lines(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [line if line.endswith("\n") else line + "\n" for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--composite", type=Path, default=Path("data/preferences/dpo_dataset.jsonl"),
                        help="composite-reward pairs (DPOTrainer format)")
    parser.add_argument("--match", type=Path, required=True,
                        help="execution-only pairs file whose pair count the subset must match")
    parser.add_argument("--out", type=Path, default=Path("data/preferences/dpo_dataset_composite_matched.jsonl"))
    parser.add_argument("--seed", type=int, default=42, help="use the same seed as DPO training")
    args = parser.parse_args()

    pairs = read_nonempty_lines(args.composite)
    target = len(read_nonempty_lines(args.match))
    if target > len(pairs):
        raise SystemExit(f"Cannot subsample {target:,} pairs from {len(pairs):,} composite pairs.")

    subset = random.Random(args.seed).sample(pairs, target)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.writelines(subset)

    print(f"{args.out}: {target:,} of {len(pairs):,} composite pairs ({target / len(pairs):.1%}), seed={args.seed}")
    print("Next: train DPO on this file with the exact config of the other two runs, evaluate it with the "
          "same harness flags, and add DPO_COMPOSITE_MATCHED_METRICS to src/evaluation/evaluation_config.py "
          "(model key 'dpo_composite_matched' in the static-analysis results) so notebook 05 picks it up.")


if __name__ == "__main__":
    main()
