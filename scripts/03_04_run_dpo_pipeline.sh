#!/bin/bash
set -e

echo "=== PHASE 3: GENERATING DPO DATASETS ==="

echo "Generating Dataset: Execution Only (Ablation)..."
uv run python -m src.preference_generation.main --reward_mode execution_only

echo "Generating Dataset: Composite (Golden)..."
uv run python -m src.preference_generation.main --reward_mode composite

echo "=== PHASE 4: DPO TRAINING ==="

echo "Training Model: Execution Only..."
uv run python -m src.training.dpo_trainer --reward_mode execution_only

echo "Training Model: Composite..."
uv run python -m src.training.dpo_trainer --reward_mode composite

echo "Training and ablation pipeline completed successfully."