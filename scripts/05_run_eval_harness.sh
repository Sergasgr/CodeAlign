#!/usr/bin/env bash

# BASE MODEL 
echo "Evaluating: Base Model"
accelerate launch tools/bigcode-evaluation-harness/main.py \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --tasks humaneval \
  --precision bf16 \
  --allow_code_execution \
  --save_generations \
  --save_generations_path data/evaluation/base_generations.json \
  --metric_output_path data/evaluation/base_metrics.json

# SFT
echo "Evaluating: SFT Model"
accelerate launch tools/bigcode-evaluation-harness/main.py \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --peft_model checkpoints/sft/final_model \
  --load_in_4bit \
  --tasks humaneval \
  --precision bf16 \
  --allow_code_execution \
  --save_generations \
  --save_generations_path data/evaluation/sft_generations.json \
  --metric_output_path data/evaluation/sft_metrics.json

# ABLATION DPO
echo "Evaluating: Ablation DPO"
accelerate launch tools/bigcode-evaluation-harness/main.py \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --peft_model checkpoints/dpo_ablation/final_model \
  --load_in_4bit \
  --tasks humaneval \
  --precision bf16 \
  --allow_code_execution \
  --save_generations \
  --save_generations_path data/evaluation/dpo_ablation_generations.json \
  --metric_output_path data/evaluation/dpo_ablation_metrics.json

# COMPOSITE REWARD DPO
echo "Evaluating: Composite Reward DPO"
accelerate launch tools/bigcode-evaluation-harness/main.py \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --peft_model checkpoints/dpo/final_model \
  --load_in_4bit \
  --tasks humaneval \
  --precision bf16 \
  --allow_code_execution \
  --save_generations \
  --save_generations_path data/evaluation/dpo_composite_generations.json \
  --metric_output_path data/evaluation/dpo_composite_metrics.json