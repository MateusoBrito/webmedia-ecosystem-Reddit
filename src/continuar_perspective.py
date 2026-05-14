import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
import os

# --- 1. CONFIGURAÇÕES (COLE SUAS 6 CHAVES AQUI) ---
API_KEYS = [
    'AIzaSyAzq86kv24HywOksRqxUDJ7T8TERhyMcXM',
    'AIzaSyBZIpJnnUEdokOyMV8Wg3iPWyWgfJjHKOY',
    'AIzaSyDTrWe0Wti2T_SPPPVSiPnyDFsx_WyE1_Q',
    'AIzaSyBPt3HzQGwRmwuEAvhp2ZSVGpWcq-LfGOc',
    'AIzaSyBLK_NhWM1wMy9rwW2HVLwq_FXJI962xfc',
    'AIzaSyB6CAS9pvCeN7YY4tc5O7z7KVc5xY4HLdU'
]

CHECKPOINT_FILE = '../data/processed/checkpoint_perspective.csv'
OUTPUT_FILE = '../data/processed/toxicidade_perspective_100k.csv'

print("📂 Lendo o arquivo de salvamento de emergência (Checkpoint)...")
df = pd.read_csv(CHECKPOINT_FILE)

# Descobre quem já tem nota e quem ainda está vazio (NaN)
mascara_faltantes = df['perspective_toxicity'].isna()
indices_faltantes = df[mascara_faltantes].index.tolist()

processados = len(df) - len(indices_faltantes)
print(f"✅ Sucesso! {processados} posts já estão salvos e seguros.")
print(f"🚀 Retomando o processamento para os {len(indices_faltantes)} posts restantes.")

# --- 2. FUNÇÃO DA THREAD ---
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

# --- 3. EXECUÇÃO EM PARALELO ---
with ThreadPoolExecutor(max_workers=len(API_KEYS)) as executor:
    
    # Só manda para o Google os textos que estão na lista de faltantes
    futuros = [executor.submit(analisar_texto, i, df.loc[i, 'text_clean']) for i in indices_faltantes]
    
    for contagem, futuro in enumerate(tqdm(as_completed(futuros), total=len(futuros), desc="Retomando API")):
        indice_original, scores = futuro.result()
        
        # Atualiza a tabela direto
        df.loc[indice_original, 'perspective_toxicity'] = scores['perspective_toxicity']
        df.loc[indice_original, 'perspective_severe_toxicity'] = scores['perspective_severe_toxicity']
        df.loc[indice_original, 'perspective_insult'] = scores['perspective_insult']
        
        # Salva o checkpoint a cada 2000 linhas processadas
        if (contagem + 1) % 2000 == 0:
            df.to_csv(CHECKPOINT_FILE, index=False)

# --- 4. FINALIZAÇÃO ---
print("\n✅ Concluído! Gerando arquivo final...")
df.to_csv(OUTPUT_FILE, index=False)

if os.path.exists(CHECKPOINT_FILE):
    os.remove(CHECKPOINT_FILE)

print(f"✨ Processo finalizado! Arquivo salvo em: {OUTPUT_FILE}")