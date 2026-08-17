#!/usr/bin/env bash
# Phase 2 — SFT (QLoRA) training + adapter merge
# Requires: Phase 1 completed (data/curated/pristine_dataset.jsonl must exist)
# GPU required: 1x 24GB VRAM (RTX 3090/4090 or equivalent)
set -euo pipefail
cd "$(dirname "$0")/.."

echo "═══ Phase 2a: SFT Training (QLoRA) ═══"
uv run python -m src.training.sft_trainer

echo ""
echo "═══ Phase 2b: Merging SFT adapter into base model ═══"
uv run python -m src.training.merge_sft

echo ""
echo "✓ Phase 2 complete."
echo "  SFT adapter: checkpoints/sft/final_model/"
echo "  Merged model: checkpoints/sft/merged_model/"
echo "  → Review W&B dashboard for training curves."
echo "  → Proceed to Phase 3: bash scripts/03_04_run_dpo_pipeline.sh"
