import os
import torch
import wandb
import argparse
import glob
import hashlib
import json
from importlib.metadata import version
from dotenv import load_dotenv
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer # type: ignore
from src.data_curation.curation_config import BASE_DIR

from src.training.dpo_config import (
    BASE_SFT_MODEL,
    TOKENIZER_MODEL,
    DPO_DS,
    DPO_OUTPUT_DIR,
    BETA,
    GRAD_ACCUMULATION_STEPS,
    LEARNING_RATE,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    MAX_SEQ_LENGTH,
    NUM_TRAIN_EPOCHS,
    PER_DEVICE_BATCH_SIZE,
    SAVE_STEPS,
    SAVE_TOTAL_LIMIT,
    SEED,
    WANDB_RUN_NAME,
)

load_dotenv()

def get_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def main():
    wandb_api_key = os.getenv("WANDB_API_KEY")
    wandb_project = os.getenv("WANDB_PROJECT")
    wandb_entity = os.getenv("WANDB_ENTITY")

    if wandb_api_key:
        wandb.login(key=wandb_api_key)
    else:
        raise ValueError("WANDB_API_KEY not found as environment variable")

    parser = argparse.ArgumentParser(description="DPO Trainer for CodeAlign")
    parser.add_argument(
        "--reward_mode",
        type=str,
        choices=["composite", "execution_only", "composite_matched"],
        default="composite",
        help="Choose which preference dataset to use for the ablation study. composite_matched = composite "
             "pairs subsampled to the execution-only pair count (scripts/make_size_matched_pairs.py)."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=LEARNING_RATE,
        help=f"Peak learning rate (default: v1.0's {LEARNING_RATE}).",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=NUM_TRAIN_EPOCHS,
        help=f"Epochs over the preference pairs (default: v1.0's {NUM_TRAIN_EPOCHS}).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Preference pairs to train on (DPOTrainer format). Default: the --reward_mode file, "
             "e.g. a *_noopfilt.jsonl from scripts/filter_noop_pairs.py.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="v1.1 runs: W&B run name and output directory checkpoints/v1.1/<run_name>/. Without it the "
             "run writes to v1.0's directory for its --reward_mode, exactly as v1.0 did."
    )
    args = parser.parse_args()

    # v1.0 defaults: dataset, output directory and W&B name per reward mode.
    if args.reward_mode == "execution_only":
        default_dataset = DPO_DS.replace(".jsonl", "_exec_only.jsonl")
        default_output = str(BASE_DIR / "checkpoints" / "dpo_ablation")
    elif args.reward_mode == "composite_matched":
        default_dataset = DPO_DS.replace(".jsonl", "_composite_matched.jsonl")
        default_output = str(BASE_DIR / "checkpoints" / "dpo_composite_matched")
    else:
        default_dataset = DPO_DS
        default_output = str(DPO_OUTPUT_DIR)
    dataset_path = args.dataset or default_dataset
    tags = ["dpo", args.reward_mode, "qwen2.5-coder"]

    if args.run_name:
        run_name = args.run_name
        output_directory = str(BASE_DIR / "checkpoints" / "v1.1" / args.run_name)
        tags.append("v1.1")
    else:
        run_name = f"{WANDB_RUN_NAME}-{args.reward_mode}"
        output_directory = default_output

    print(f"Run: {run_name} | pairs: {dataset_path} | lr={args.learning_rate} | "
          f"epochs={args.num_train_epochs} | output: {output_directory}")

    dataset_raw = load_dataset(
        "json",
        data_files=dataset_path,
        split="train"
    )

    def format_to_conversational(example):
        return {
            "prompt": [{"role": "user", "content": example["prompt"]}],
            "chosen": [{"role": "assistant", "content": example["chosen"]}],
            "rejected": [{"role": "assistant", "content": example["rejected"]}]
        }

    dataset = dataset_raw.filter(
        lambda x: bool(x["chosen"].strip()) and bool(x["rejected"].strip())
    ).map(format_to_conversational)

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_MODEL,
        use_fast=True,
        padding_side="right"
    )
    tokenizer.pad_token = tokenizer.eos_token

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_SFT_MODEL,
        quantization_config=quantization_config,
        device_map="auto",
        use_cache=False
    )

    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=LORA_DROPOUT,
        task_type="CAUSAL_LM"
    )

    training_args = DPOConfig(
        output_dir=output_directory,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION_STEPS,
        bf16=True,
        learning_rate=args.learning_rate,
        max_length=MAX_SEQ_LENGTH,
        logging_steps=10,
        report_to="wandb",
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        beta=BETA,
        dataset_num_proc=4,
        seed=SEED,
    )

    os.environ["WANDB_LOG_MODEL"] = "false"

    wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        name=run_name,
        tags=tags,
        resume="allow"
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    checkpoints = glob.glob(f"{output_directory}/checkpoint-*")
    resume_from_checkpoint = True if checkpoints else False

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    final_dir = f"{output_directory}/final_model"
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    run_config = {
        "run_name": run_name,
        "reward_mode": args.reward_mode,
        "learning_rate": training_args.learning_rate,
        "num_train_epochs": training_args.num_train_epochs,
        "dataset_path": dataset_path,
        "dataset_sha256": get_sha256(dataset_path),
        "dataset_rows": len(dataset_raw),
        "final_pair_count": len(dataset),
        "global_step": trainer.state.global_step,
        "base_model": BASE_SFT_MODEL,
        "beta": training_args.beta,
        "loss_type": training_args.loss_type,
        "lr_scheduler_type": training_args.lr_scheduler_type,
        "warmup_steps": training_args.warmup_steps,
        "batch_size": training_args.per_device_train_batch_size,
        "grad_accumulation": training_args.gradient_accumulation_steps,
        "max_length": training_args.max_length,
        "optim": training_args.optim,
        "seed": training_args.seed,
        "lora_r": peft_config.r,
        "lora_alpha": peft_config.lora_alpha,
        "lora_dropout": peft_config.lora_dropout,
        "target_modules": sorted(peft_config.target_modules),
        "quantization": "nf4, double quant, bf16 compute",
        "versions": {pkg: version(pkg) for pkg in ("trl", "transformers", "peft", "torch", "bitsandbytes")},
    }

    with open(os.path.join(final_dir, "run_config.json"), "w", encoding='utf-8') as f:
        json.dump(run_config, f, indent=4, default=str)

    print(f"Run config saved to {os.path.join(final_dir, 'run_config.json')}")

    wandb.finish()

if __name__ == "__main__":
    main()