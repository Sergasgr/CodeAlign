import os
import torch
import wandb 
import argparse
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
        choices=["composite", "execution_only"], 
        default="composite",
        help="Choose which preference dataset to use for the ablation study."
    )
    args = parser.parse_args()
    
    if args.reward_mode == "execution_only":
        dataset_path = DPO_DS.replace(".jsonl", "_exec_only.jsonl") 
        run_name = f"{WANDB_RUN_NAME}-ablation-exec-only"
        tags = ["dpo", "preference-dataset", "execution-only", "qwen2.5-coder"]
    else:
        dataset_path = DPO_DS
        run_name = f"{WANDB_RUN_NAME}-composite"
        tags = ["dpo", "preference-dataset", "composite-reward", "qwen2.5-coder"]
        
    dataset = load_dataset(
        "json",
        data_files=dataset_path, 
        split="train"
    ) 

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

    output_directory = str(DPO_OUTPUT_DIR) if args.reward_mode == "composite" else str(BASE_DIR / "checkpoints" / "dpo_ablation")

    training_args = DPOConfig( 
        output_dir=output_directory, 
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,         
        gradient_accumulation_steps=GRAD_ACCUMULATION_STEPS,          
        bf16=True,                             
        learning_rate=LEARNING_RATE,                  
        max_length=MAX_SEQ_LENGTH,                  
        logging_steps=10,                     
        report_to="wandb",                     
        gradient_checkpointing=True,            
        optim="paged_adamw_8bit",
        beta=BETA, 
        dataset_num_proc=4,
        seed=SEED,
    )

    wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        name=run_name,
        tags=tags
    )

    trainer = DPOTrainer(
        model=model,
        train_dataset=dataset,  
        peft_config=peft_config,
        processing_class=tokenizer, 
        args=training_args,
    )

    trainer.train()

    final_dir = f"{output_directory}/final_model"
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    wandb.finish()

if __name__ == "__main__":
    main()