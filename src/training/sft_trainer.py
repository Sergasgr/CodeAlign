import wandb 
import os
from dotenv import load_dotenv
from datasets import load_dataset
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
import torch
from peft import LoraConfig
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer

from configs.config import (
    DS_PATH, TOKENIZER_MODEL, SFT_MODEL, SFT_OUTPUT_DIR,
    LORA_R, LORA_ALPHA, LORA_DROPOUT, LEARNING_RATE, 
    PER_DEVICE_BATCH_SIZE, GRAD_ACCUMULATION_STEPS, MAX_SEQ_LENGTH,
    SAVE_STEPS, SAVE_TOTAL_LIMIT, WANDB_RUN_NAME
)

load_dotenv()

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
) # num_proc?

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

config = LoraConfig( 
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=LORA_DROPOUT,
    task_type="CAUSAL_LM"
)

training_args = SFTConfig( 
    output_dir=SFT_OUTPUT_DIR, 
    save_strategy="steps",
    save_steps=SAVE_STEPS, # o 100
    save_total_limit=SAVE_TOTAL_LIMIT, # o 3
    per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,         
    gradient_accumulation_steps=GRAD_ACCUMULATION_STEPS,          
    bf16=True,                             
    learning_rate=LEARNING_RATE,                  
    max_seq_length=MAX_SEQ_LENGTH, # No parameter named "max_seq_length"                   
    logging_steps=10,                     
    report_to="wandb",                     
    gradient_checkpointing=True,            
    optim="paged_adamw_8bit"                
)

wandb.init(
    project=wandb_project,
    entity=wandb_entity,
    name=WANDB_RUN_NAME,
    tags=["sft", "pristine-dataset", "qwen2.5-coder"]
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,  
    peft_config=config,
    tokenizer=tokenizer, # No parameter named "tokenizer"
    args=training_args,
)

trainer.train()

trainer.save_model(f"{SFT_OUTPUT_DIR}/final_model")
wandb.finish()