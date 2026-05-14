import pandas as pd
import time
import os
from googleapiclient import discovery
from googleapiclient.errors import HttpError
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# --- CONFIGURAÇÕES DE ARQUIVO ---
ARQUIVO_TOTAL = '../data/processed/preprocess_text.csv' 
ARQUIVO_JA_PRONTO = '../data/processed/toxicidade_perspective_100k.csv'
ARQUIVO_NOVO_PROGRESSO = '../data/processed/toxicidade_perspective_PROGRESSO.csv'

# Salva no disco a cada 300 posts (não afeta a barra visual, só a gravação do arquivo)
SALVAR_A_CADA = 300 

# COLOQUE SUAS 6 CHAVES REAIS AQUI
CHAVES_API = [
    'AIzaSyAzq86kv24HywOksRqxUDJ7T8TERhyMcXM',
    'AIzaSyBZIpJnnUEdokOyMV8Wg3iPWyWgfJjHKOY',
    'AIzaSyDTrWe0Wti2T_SPPPVSiPnyDFsx_WyE1_Q',
    'AIzaSyBPt3HzQGwRmwuEAvhp2ZSVGpWcq-LfGOc',
    'AIzaSyBLK_NhWM1wMy9rwW2HVLwq_FXJI962xfc',
    'AIzaSyB6CAS9pvCeN7YY4tc5O7z7KVc5xY4HLdU'
]

WORKERS = len(CHAVES_API)

# --- FUNÇÕES PARA ACESSAR A API DO GOOGLE ---
def get_client(key):
    return discovery.build(
        "commentanalyzer", "v1alpha1", developerKey=key,
        discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
        static_discovery=False,
    )

def processar_post(dados):
    id_post, texto, chave = dados
    client = get_client(chave)
    texto_cortado = str(texto)[:3000]
    
    while True: # Tenta até conseguir, respeitando o limite do Google
        try:
            request = {'comment': {'text': texto_cortado}, 'requestedAttributes': {'TOXICITY': {}}}
            response = client.comments().analyze(body=request).execute()
            score = response['attributeScores']['TOXICITY']['summaryScore']['value']
            return {'id': id_post, 'perspective_toxicity': score}
        except HttpError as e:
            if e.resp.status == 429: # Rate Limit
                time.sleep(2)
            else:
                return {'id': id_post, 'perspective_toxicity': None}
        except Exception:
            return {'id': id_post, 'perspective_toxicity': None}

# --- LÓGICA DE VERIFICAÇÃO ---
print("🔍 Verificando progresso anterior...")

df_antigo = pd.read_csv(ARQUIVO_JA_PRONTO, usecols=['id'])
ids_concluidos = set(df_antigo['id'].unique())

if os.path.exists(ARQUIVO_NOVO_PROGRESSO):
    df_progresso = pd.read_csv(ARQUIVO_NOVO_PROGRESSO, usecols=['id'])
    ids_concluidos.update(df_progresso['id'].unique())

print(f"✅ Total de posts ignorados (já processados): {len(ids_concluidos)}")

df_total = pd.read_csv(ARQUIVO_TOTAL)
df_todo = df_total[~df_total['id'].isin(ids_concluidos)].reset_index(drop=True)
total_faltante = len(df_todo)
print(f"🚀 Faltam processar: {total_faltante} posts.")

# --- PREPARAÇÃO DA ESTEIRA (GERADOR) ---
def gerador_tarefas():
    for i, row in enumerate(df_todo.itertuples()):
        chave_designada = CHAVES_API[i % len(CHAVES_API)]
        yield (row.id, row.text_clean, chave_designada)

# --- INÍCIO DO PROCESSAMENTO PARALELO ---
print(f"⚡ Iniciando Multithreading com {WORKERS} conexões simultâneas...")

batch_atual = []

try:
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        # AQUI É ONDE A MÁGICA ACONTECE: executor.map processa em paralelo, mas o tqdm conta de 1 em 1
        for resultado in tqdm(executor.map(processar_post, gerador_tarefas()), total=total_faltante, desc="Analisando Posts"):
            if resultado:
                batch_atual.append(resultado)
            
            # Quando a sacola enche, salva no disco (não atrapalha a contagem da barra)
            if len(batch_atual) >= SALVAR_A_CADA:
                df_save = pd.DataFrame(batch_atual)
                df_save.to_csv(ARQUIVO_NOVO_PROGRESSO, mode='a', index=False, 
                               header=not os.path.exists(ARQUIVO_NOVO_PROGRESSO))
                batch_atual = [] # Esvazia a sacola

except KeyboardInterrupt:
    print("\n🛑 Você interrompeu o script. Salvando o que foi feito até agora...")

# Salva o que restou na sacola caso você pare o código antes de bater os 300
if batch_atual:
    df_save = pd.DataFrame(batch_atual)
    df_save.to_csv(ARQUIVO_NOVO_PROGRESSO, mode='a', index=False, 
                   header=not os.path.exists(ARQUIVO_NOVO_PROGRESSO))

print("🏁 Script finalizado ou pausado.")