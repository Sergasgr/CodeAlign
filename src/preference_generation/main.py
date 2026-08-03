import json
from preference_orchestrator import PreferenceOrchestrator
from src.data_curation.curation_config import DS_PATH
from src.preference_generation.preference_generation_config import DPO_DS

def main():
    orchestrator = PreferenceOrchestrator()
    # Deberás definir la ruta de entrada (¿De dónde sacamos los prompts? Podría ser una muestra de tu pristine_dataset.jsonl o un conjunto de evaluación diferente) -> utilizo pristine dataset?????. Hago split de validación/prueba de CommitPackFT) o extrigo M muestras aleatorias de tu dataset actual?
    with open(DS_PATH, 'r', encoding='utf-8') as ds, open(DPO_DS, 'w', encoding='utf-8') as dpo_f:
        for line in ds: # COMO LO ENTRENO (Mirar arriba)
            prompt = json.loads(line)["messages"][0]["content"]
            language = json.loads(line)["metadata"]["language"]
            preference_pair_dict = orchestrator.create_preference_pair(prompt, language)
            if preference_pair_dict:
                dpo_f.write(json.dumps(preference_pair_dict, ensure_ascii=False) + "\n")
        
if __name__ == "__main__":
    main()
    
"""
import json
import random
from tqdm import tqdm
from src.preference_generation.preference_orchestrator import PreferenceOrchestrator
from src.data_curation.curation_config import DS_PATH
from src.preference_generation.preference_generation_config import DPO_DS

def main():
    # 1. Configuración de muestreo
    NUM_PROMPTS_TO_GENERATE = 3000 # Ajusta esto según tus horas de GPU disponibles
    
    # 2. Cargar y muestrear datos
    print(f"Cargando dataset base desde {DS_PATH}...")
    all_data = []
    with open(DS_PATH, 'r', encoding='utf-8') as ds:
        for line in ds:
            all_data.append(json.loads(line))
            
    # Tomar una muestra aleatoria para no generar los 100k+ de golpe
    sampled_data = random.sample(all_data, min(NUM_PROMPTS_TO_GENERATE, len(all_data)))
    
    # 3. Inicializar orquestador (carga el modelo en GPU)
    print("Inicializando modelo SFT y Sandbox Docker...")
    orchestrator = PreferenceOrchestrator()
    
    # 4. Bucle principal de generación
    print(f"Generando pares de preferencias para {len(sampled_data)} prompts...")
    
    # Abrimos en modo 'w' (escritura nueva)
    with open(DPO_DS, 'w', encoding='utf-8') as dpo_f:
        # tqdm nos da una barra de progreso visual con tiempo estimado (ETA)
        for data in tqdm(sampled_data, desc="Generando DPO pairs"):
            
            # Extraer prompt y lenguaje según el formato de prompts.py
            prompt = data["messages"][0]["content"] 
            language = data["metadata"]["language"]
            
            # La magia ocurre aquí
            preference_pair_dict = orchestrator.create_preference_pair(prompt, language)
            
            # Si pasó los filtros, lo guardamos
            if preference_pair_dict:
                dpo_f.write(json.dumps(preference_pair_dict, ensure_ascii=False) + "\n")
                dpo_f.flush() # Fuerza el guardado en disco al instante por si se corta la ejecución

if __name__ == "__main__":
    main()
"""