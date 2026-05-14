import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
import os

# --- 1. CONFIGURAÇÕES (COLE SUAS 6 CHAVES) ---
API_KEYS = [
    'AIzaSyAzq86kv24HywOksRqxUDJ7T8TERhyMcXM',
    'AIzaSyBZIpJnnUEdokOyMV8Wg3iPWyWgfJjHKOY',
    'AIzaSyDTrWe0Wti2T_SPPPVSiPnyDFsx_WyE1_Q',
    'AIzaSyBPt3HzQGwRmwuEAvhp2ZSVGpWcq-LfGOc',
    'AIzaSyBLK_NhWM1wMy9rwW2HVLwq_FXJI962xfc',
    'AIzaSyB6CAS9pvCeN7YY4tc5O7z7KVc5xY4HLdU'
]

INPUT_FILE = '../data/processed/preprocess_text.csv' 
OUTPUT_FILE = '../data/processed/toxicidade_perspective_100k.csv'
CHECKPOINT_FILE = '../data/processed/checkpoint_perspective.csv'
TEXT_COLUMN = 'text_clean'
LIMIT_ROWS = 100000 

print(f"📂 Lendo as primeiras {LIMIT_ROWS} linhas de: {INPUT_FILE}")
df = pd.read_csv(INPUT_FILE, nrows=LIMIT_ROWS)
df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna('')
textos = df[TEXT_COLUMN].tolist()

# --- A CORREÇÃO ESTÁ AQUI ---
# Em vez de [None], criamos a lista já com a estrutura certa para o Pandas não quebrar
resultados = [{'perspective_toxicity': None, 'perspective_severe_toxicity': None, 'perspective_insult': None} for _ in range(len(textos))]

# --- 2. FUNÇÃO QUE CADA THREAD VAI EXECUTAR ---
def analisar_texto(i, texto):
    if not str(texto).strip():
        return i, {'perspective_toxicity': 0.0, 'perspective_severe_toxicity': 0.0, 'perspective_insult': 0.0}

    chave = API_KEYS[i % len(API_KEYS)]
    url = f"https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze?key={chave}"
    
    payload = {
        'comment': {'text': str(texto)},
        'languages': ['en'],
        'requestedAttributes': {'TOXICITY': {}, 'SEVERE_TOXICITY': {}, 'INSULT': {}}
    }

    tentativas = 0
    while tentativas < 3:
        try:
            response = requests.post(url, json=payload, timeout=10)
            
            if response.status_code == 200:
                dados = response.json()
                scores = {
                    'perspective_toxicity': dados['attributeScores']['TOXICITY']['summaryScore']['value'],
                    'perspective_severe_toxicity': dados['attributeScores']['SEVERE_TOXICITY']['summaryScore']['value'],
                    'perspective_insult': dados['attributeScores']['INSULT']['summaryScore']['value']
                }
                time.sleep(1.0)
                return i, scores
                
            elif response.status_code == 429: 
                time.sleep(3) 
                tentativas += 1
            else:
                time.sleep(2)
                tentativas += 1
                
        except Exception:
            time.sleep(2)
            tentativas += 1

    return i, {'perspective_toxicity': None, 'perspective_severe_toxicity': None, 'perspective_insult': None}

# --- 3. EXECUÇÃO EM PARALELO (MULTITHREADING) ---
print(f"🚀 Iniciando {len(textos)} posts com {len(API_KEYS)} chaves em PARALELO.")

with ThreadPoolExecutor(max_workers=len(API_KEYS)) as executor:
    
    futuros = [executor.submit(analisar_texto, i, texto) for i, texto in enumerate(textos)]
    
    for contagem, futuro in enumerate(tqdm(as_completed(futuros), total=len(futuros), desc="Perspective API")):
        indice_original, scores = futuro.result()
        resultados[indice_original] = scores
        
        # Checkpoint a cada 2000 linhas processadas
        if (contagem + 1) % 2000 == 0:
            df_temp = pd.DataFrame(resultados)
            df_checkpoint = pd.concat([df.reset_index(drop=True), df_temp], axis=1)
            df_checkpoint.to_csv(CHECKPOINT_FILE, index=False)

# --- 4. FINALIZAÇÃO ---
print("\n✅ Concluído! Gerando arquivo final...")
df_perspective = pd.DataFrame(resultados)
df_final = pd.concat([df.reset_index(drop=True), df_perspective], axis=1)

df_final.to_csv(OUTPUT_FILE, index=False)

if os.path.exists(CHECKPOINT_FILE):
    os.remove(CHECKPOINT_FILE)

print(f"✨ Processo finalizado! Arquivo salvo em: {OUTPUT_FILE}")