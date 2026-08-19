import os
import json
import torch
import wandb
from typing import Any
from dotenv import load_dotenv
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer

from src.data_curation.curation_config import DS_PATH
from src.preference_generation.preference_orchestrator import PreferenceOrchestrator
from src.training.grpo_config import ( # Asumiendo que guardaste el config como grpo_config.py
    BASE_SFT_MODEL,
    TOKENIZER_MODEL,
    GRPO_OUTPUT_DIR,
    NUM_GENERATIONS,
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

"""
def composite_reward_func(prompts: list[str], completions: list[str], language: list[str], **kwargs) -> list[float]:
    scores = []
    for prompt, completion, lang in zip(prompts, completions, language): 
        code = completion[0]["content"] if isinstance(completion, list) else completion 1 # Argument of type "Literal['content']" cannot be assigned to parameter "key" of type "SupportsIndex | slice[SupportsIndex | None, SupportsIndex | None, SupportsIndex | None]" in function "__getitem__"
        eval_result = orchestrator.reward_score(code, lang)
        scores.append(float(eval_result["score"]))
    return scores
"""

def composite_reward_func(prompts: list[str], completions: list[Any], **kwargs) -> list[float]:
    language = kwargs.get("language", [])
    scores = []
    for prompt, completion, lang in zip(prompts, completions, language): 
        code = completion[0]["content"] if isinstance(completion, list) else completion 
        eval_result = orchestrator.reward_score(code, lang)
        scores.append(float(eval_result["score"]))
    return scores

def prepare_grpo_dataset(path: str) -> Dataset:
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line)
            data.append({
                "prompt": sample["messages"][0]["content"], 
                "language": sample["metadata"]["language"] 
            })
    return Dataset.from_list(data)

def main():
    wandb_api_key = os.getenv("WANDB_API_KEY")
    wandb_project = os.getenv("WANDB_PROJECT") 
    wandb_entity = os.getenv("WANDB_ENTITY")

    if wandb_api_key:
        wandb.login(key=wandb_api_key)
    else:
        raise ValueError("WANDB_API_KEY not found as environment variable")
    
    dataset = prepare_grpo_dataset(str(DS_PATH))
    
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
    
    training_args = GRPOConfig( 
        output_dir=str(GRPO_OUTPUT_DIR), 
        num_generations=NUM_GENERATIONS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,         
        gradient_accumulation_steps=GRAD_ACCUMULATION_STEPS,          
        bf16=True,                             
        learning_rate=LEARNING_RATE,                  
        max_completion_length=MAX_SEQ_LENGTH,            
        logging_steps=10,                     
        report_to="wandb",                     
        gradient_checkpointing=True,            
        optim="paged_adamw_8bit",
        beta=BETA, 
        dataloader_num_workers=4, #dataset_num_proc=4, No parameter named "dataset_num_proc" PylancereportCallIssue
        use_vllm=False, 
        seed=SEED,
    )
    
    wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        name=WANDB_RUN_NAME,
        tags=["grpo", "rl", "composite-reward", "qwen2.5-coder"]
    )
    
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
        reward_funcs=[composite_reward_func], # Argument of type "list[(prompts: list[str], completions: list[str], language: list[str], **kwargs: Unknown) -> list[float]]" cannot be assigned to parameter "reward_funcs" of type "RewardFunc | list[RewardFunc] | None" in function "__init__". Type "(prompts: list[str], completions: list[str], language: list[str], **kwargs: Unknown) -> list[float]" is not assignable to type "RewardFunc"
    )
    trainer.train()
    
    final_dir = f"{GRPO_OUTPUT_DIR}/final_model"
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    
    wandb.finish()
    
if __name__ == "__main__":
    orchestrator = PreferenceOrchestrator()
    main()