import os
import torch
import wandb 
from dotenv import load_dotenv
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer # type: ignore

from src.data_curation.curation_config import DS_PATH
from src.training.sft_config import (
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
    SFT_MODEL,
    SFT_OUTPUT_DIR,
    TOKENIZER_MODEL,
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

    dataset = load_dataset(
        "json",
        data_files=str(DS_PATH), 
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
        SFT_MODEL,
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

    training_args = SFTConfig( 
        output_dir=str(SFT_OUTPUT_DIR), 
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
        seed=SEED,
    )

    os.environ["WANDB_LOG_MODEL"] = "false"
    
    wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        name=WANDB_RUN_NAME,
        tags=["sft", "pristine-dataset", "qwen2.5-coder"],
        resume="allow"
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,  
        peft_config=peft_config,
        processing_class=tokenizer, 
        args=training_args,
    )

    import glob
    checkpoints = glob.glob(f"{SFT_OUTPUT_DIR}/checkpoint-*")
    resume_from_checkpoint = True if checkpoints else False

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    final_dir = f"{SFT_OUTPUT_DIR}/final_model"
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    wandb.finish()

if __name__ == "__main__":
    main()