"""
Lo que hará: Un pequeño script de Python que busque en tus resultados unos 5 ejemplos donde tanto el modelo de Ablación como el Compuesto hayan resuelto el problema de HumanEval, pero el modelo Compuesto tenga mucha menos complejidad.

Salida: Imprimirá el código de ambos modelos lado a lado en formato Markdown para que puedas copiarlo y pegarlo directamente en tu README.md.
"""
import json
from src.evaluation.evaluation_config import (
    DPO_ABLATION_EVALUATED_MODEL,
    DPO_COMPOSITE_REWARD_EVALUATED_MODEL,
    OUTPUT_MD,
    TARGET_SAMPLES,
    COMPLEXITY_GAP
)
from src.data_curation.validators import linter_check

with open(DPO_ABLATION_EVALUATED_MODEL, 'r', encoding='utf-8') as f1:
    dpo_ablation = json.load(f1)
    
with open(DPO_COMPOSITE_REWARD_EVALUATED_MODEL, 'r', encoding='utf-8') as f2:
    dpo_composite_reward = json.load(f2)
    
total_samples = 0
    
with open(OUTPUT_MD, 'w', encoding='utf-8') as f:
    f.write("## Quality samples: Ablation DPO vs. Composite Reward DPO\n\n")
    for idx, (dpo_abl, dpo_comp) in enumerate(zip(dpo_ablation, dpo_composite_reward)):
        code_abl = dpo_abl[0]
        code_comp = dpo_comp[0]
        result_abl = linter_check(code_abl, "python") #deberia haber soporte multilenguaje hay que cambiar
        result_comp = linter_check(code_comp, "python")
        complexity_abl = result_abl.get("complexity") or 0
        complexity_comp = result_comp.get("complexity") or 0
        if complexity_abl - complexity_comp >= COMPLEXITY_GAP:
            total_samples += 1
            markdown_block = f"""### Problema HumanEval #{idx}
            **Complejidad:** Ablación ({complexity_abl}) vs. Compuesto ({complexity_comp})

            **DPO Ablación (Spaghetti Code):**
            ```python
            {code_abl.strip()}
            {code_comp.strip()}
            """
            f.write(markdown_block)
            if total_samples == TARGET_SAMPLES: 
                break
            
print(f"Success: It has been exctracted {total_samples} quality samples in {OUTPUT_MD}")