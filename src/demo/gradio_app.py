import gradio as gr
from src.demo.models_initializer import base_model, model, tokenizer
from src.demo.demo_config import TEMPERATURE_DEMO, TOP_P_DEMO
from src.data_curation.validators import linter_check, parse_code

""" INCLUIR EN README
Sin embargo, antes de tirar líneas, tenemos que resolver un problema crítico de ingeniería: cargar tres modelos de 7B en la GPU al mismo tiempo te dará un error de falta de memoria (OOM), incluso en una RTX 3090.

Aquí tienes cómo vamos a estructurar esta fase para que sea eficiente y profesional.
1. El truco de Hardware: Intercambio de Adaptadores LoRA

Como entrenaste con QLoRA (Fase 2 y Fase 4), no tienes tres modelos gigantes; tienes un modelo base y dos pequeños adaptadores (SFT y DPO).
En lugar de cargar tres modelos distintos, cargarás el modelo base una sola vez y usarás la librería peft para activar y desactivar los adaptadores al vuelo.

    Modelo Base: Desactivas todos los adaptadores (model.disable_adapters()).

    Modelo SFT: Activas el adaptador SFT (model.set_adapter("sft")).

    Modelo DPO: Activas el adaptador DPO (model.set_adapter("dpo")).

Esto demuestra un conocimiento profundo sobre inferencia eficiente en hardware.
"""

def execute_model(prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        do_sample=True,      
        temperature=TEMPERATURE_DEMO,
        top_p=TOP_P_DEMO,      
        max_new_tokens=512, # Limitado para que la demo sea rápida
        pad_token_id=tokenizer.pad_token_id
    )
    input_length = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][input_length:]
    raw_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return parse_code(raw_text)

def format_metrics(metrics: dict) -> str:
    return f"**Complejidad:** {metrics['complexity']} | **Lint Errors:** {metrics['lint_errors']}"

def generate_comparison(prompt: str):
    # --- 1. BASE MODEL ---
    model.disable_adapters() # Object of type "Tensor" is not callable: Attribute "__call__" is unknown
    base_code = execute_model(prompt) 
    base_metrics = linter_check(base_code, "python")
    
    # --- 2. SFT MODEL ---
    model.set_adapter("sft")
    sft_code = execute_model(prompt)
    sft_metrics = linter_check(sft_code, "python")
    
    model.set_adapter("dpo")
    dpo_code = execute_model(prompt)
    dpo_metrics = linter_check(dpo_code, "python")
    
    return (
        base_code, format_metrics(base_metrics),
        sft_code, format_metrics(sft_metrics),
        dpo_code, format_metrics(dpo_metrics)
    )
    
def main(): # TRADUCIR
    theme = gr.themes.Soft(primary_hue="blue") # "themes" is not exported from module "gradio"PylancereportPrivateImportUsage
    
    with gr.Blocks(theme=theme, title="CodeAlign Demo") as demo:
        gr.Markdown("# 🚀 CodeAlign: DPO vs SFT vs Base")
        gr.Markdown("Escribe un problema de programación y compara empíricamente cómo el **Composite Reward** reduce el *spaghetti code* frente a un modelo base o SFT.")
        
        inp = gr.Textbox(lines=4, label="Instrucción de código", placeholder="Escribe aquí tu prompt...")
        btn = gr.Button("Generar y Comparar", variant="primary") 
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Modelo Base")
                out_base_code = gr.Code(language="python", label="Código Generado")
                out_base_metrics = gr.Markdown()
                
            with gr.Column():
                gr.Markdown("### Modelo SFT")
                out_sft_code = gr.Code(language="python", label="Código Generado")
                out_sft_metrics = gr.Markdown()
                
            with gr.Column():
                gr.Markdown("### Modelo DPO (Composite Reward) ⭐")
                out_dpo_code = gr.Code(language="python", label="Código Generado")
                out_dpo_metrics = gr.Markdown()
                
        btn.click( # Cannot access attribute "click" for class "Button" Attribute "click" is unknown
            fn=generate_comparison, 
            inputs=[inp], 
            outputs=[
                out_base_code, out_base_metrics,
                out_sft_code, out_sft_metrics,
                out_dpo_code, out_dpo_metrics
            ]
        )
        
    demo.launch(share=False) # Pon share=True si quieres un link público temporal

if __name__ == "__main__":
    main()