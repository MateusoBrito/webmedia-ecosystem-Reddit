import sys
import os

# --- O PULO DO GATO PARA O PYTHON ACHAR AS PASTAS ---
raiz_do_projeto = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(raiz_do_projeto)

import pandas as pd
import time
import gc
from googleapiclient import discovery
from googleapiclient.errors import HttpError
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from src.utils.paths import ROOT
from src.utils.data_loader import load_preprocessed_data

# --- CONFIGURAÇÕES DE ARQUIVO ---
# NOME NOVO PARA NÃO MISTURAR COM O ANTIGO
ARQUIVO_NOVO_PROGRESSO = ROOT / "data" / "processed" / "toxicidade_perspective_COMPLETO.csv"
SALVAR_A_CADA = 300 

# --- COLE AQUI AS SUAS 20 CHAVES ---
CHAVES_API = [
    'AIzaSyAzq86kv24HywOksRqxUDJ7T8TERhyMcXM',
    'AIzaSyBZIpJnnUEdokOyMV8Wg3iPWyWgfJjHKOY',
    'AIzaSyDTrWe0Wti2T_SPPPVSiPnyDFsx_WyE1_Q',
    'AIzaSyBPt3HzQGwRmwuEAvhp2ZSVGpWcq-LfGOc',
    'AIzaSyBLK_NhWM1wMy9rwW2HVLwq_FXJI962xfc',
    'AIzaSyB6CAS9pvCeN7YY4tc5O7z7KVc5xY4HLdU',
    'AIzaSyDtt4nzoJXd0IWk2m8aSS2WrilV1thi10Q',
    'AIzaSyCFmOnvOjAuTQH1GoVrcCtnNezriNvr4c0',
    'AIzaSyCgcR_L1VgQ8H4AW1FkhCxjiRy4tRI1GWw',
    'AIzaSyCvtJdCoEgru4At7_Ib12aPfw9s19lHVYk',
    'AIzaSyATXcuZsUhMXVlRzmMao34Od9siMID823Y',
    'AIzaSyDTn8Oeq2lKuIpzEcI8ScPhSfo-ObDpMCg',
    'AIzaSyCnJe--UZ9rLRdhtE5LTIDDkh4wawlx7tU',
    'AIzaSyCFobvh5Bdj7Qnpl2c6cjMNvgMqDLnI0qg',
    'AIzaSyAS-VyYPBTYjXXFB_T5eriwBtVBWZhsZ3c',
    'AIzaSyCCnU1uIWf5g0uZDPL-j2Bhs0hEHODmmaQ',
    'AIzaSyAk7GHAw9Takoe-Y-Z_vrea9t7GgeRJDxI',
    'AIzaSyDBhLzNLiZewfOSkocg72v6zmYsbqw535E'
    # COLOQUE AS OUTRAS AQUI
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
    
    while True:
        try:
            # Pedindo o pacote completo de análises!
            request = {
                'comment': {'text': texto_cortado}, 
                'requestedAttributes': {
                    'TOXICITY': {},
                    'SEVERE_TOXICITY': {},
                    'IDENTITY_ATTACK': {},
                    'INSULT': {},
                    'PROFANITY': {},
                    'THREAT': {}
                }
            }
            response = client.comments().analyze(body=request).execute()
            scores = response['attributeScores']
            
            return {
                'id': id_post, 
                'perspective_toxicity': scores['TOXICITY']['summaryScore']['value'],
                'severe_toxicity': scores['SEVERE_TOXICITY']['summaryScore']['value'],
                'identity_attack': scores['IDENTITY_ATTACK']['summaryScore']['value'],
                'insult': scores['INSULT']['summaryScore']['value'],
                'profanity': scores['PROFANITY']['summaryScore']['value'],
                'threat': scores['THREAT']['summaryScore']['value']
            }
            
        except HttpError as e:
            if e.resp.status == 429: # Rate Limit
                time.sleep(2)
            else:
                return {'id': id_post, 'perspective_toxicity': None, 'severe_toxicity': None, 'identity_attack': None, 'insult': None, 'profanity': None, 'threat': None}
        except Exception:
            return {'id': id_post, 'perspective_toxicity': None, 'severe_toxicity': None, 'identity_attack': None, 'insult': None, 'profanity': None, 'threat': None}

# --- LÓGICA DE VERIFICAÇÃO INTELIGENTE ---
print("⏳ Carregando APENAS a base filtrada de posts válidos (Parquet)...")
df_total = load_preprocessed_data(only_valid_ids=True, columns=['id', 'text_clean'])
print(f"📊 Base filtrada carregada! Total de posts válidos: {len(df_total)}")

print("🔍 Verificando o que já foi avaliado anteriormente...")
ids_concluidos = set()

if ARQUIVO_NOVO_PROGRESSO.exists():
    df_progresso = pd.read_csv(ARQUIVO_NOVO_PROGRESSO, usecols=['id'])
    ids_concluidos.update(df_progresso['id'].unique())
    print(f"✅ Identificados {len(ids_concluidos)} posts já salvos com sucesso.")

df_todo = df_total[~df_total['id'].isin(ids_concluidos)].reset_index(drop=True)
total_faltante = len(df_todo)
print(f"🚀 Cruzamento feito! Faltam processar: {total_faltante} posts.")

# ==========================================
# LIMPANDO A MEMÓRIA PARA EVITAR O ERRO "KILLED"
del df_total
gc.collect() 
print("🧹 Memória RAM liberada com sucesso!")
# ==========================================

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
        for resultado in tqdm(executor.map(processar_post, gerador_tarefas()), total=total_faltante, desc="Analisando Posts"):
            if resultado:
                batch_atual.append(resultado)
            
            if len(batch_atual) >= SALVAR_A_CADA:
                df_save = pd.DataFrame(batch_atual)
                df_save.to_csv(ARQUIVO_NOVO_PROGRESSO, mode='a', index=False, 
                               header=not ARQUIVO_NOVO_PROGRESSO.exists())
                batch_atual = []

except KeyboardInterrupt:
    print("\n🛑 Você interrompeu o script. Salvando o lote atual...")

if batch_atual:
    df_save = pd.DataFrame(batch_atual)
    df_save.to_csv(ARQUIVO_NOVO_PROGRESSO, mode='a', index=False, 
                   header=not ARQUIVO_NOVO_PROGRESSO.exists())

print("🏁 Script finalizado ou pausado.")