"""
Este .py seleciona os subreddits que foram alcançados a partir dos subreddits com maior ranking de conteúdo da
""" 

#%%
import sys
import os

sys.path.append('/home/mateus/reddit')
from src.utils.data_loader import load_preprocessed_data,load_ranking_subreddits,load_toxicity,ROOT
import pandas as pd
from pathlib import Path

df = load_preprocessed_data()
df_topic = pd.read_parquet(Path(ROOT/"data"/"processed"/"post_topics_1.parquet"))
df_ranking = load_ranking_subreddits()
# %%
print(f"Colunas do df original: {df.columns}")
print(f"Colunas do df dos topicos: {df_topic.columns}")
print(f"Colunas do df do ranking: {df_ranking.columns}")
#%%
print(f"Tamanho do df original: {len(df)}")
print(f"Tamanho do df dos topicos: {len(df_topic)}")
print(f"Tamanho do df do ranking: {len(df_ranking)}")
#%%
df_full = pd.merge(df,df_topic,on="id",how="inner")
df_full = pd.merge(df_full, df_ranking, on="subreddit",how="inner")
#%%
print(f"Colunas do df resultante: {df_full.columns}")
print(f"Tamanho do df resultante: {len(df_full)}")

TOPIC_TO_MACROTOPIC = {
    0: "Ruído, Humor e Interações Genéricas",
    1: "Relações Afetivas, Familiares e Vida Cotidiana",
    2: "Entretenimento, Cultura e Produção Criativa",
    3: "Ruído, Humor e Interações Genéricas",
    4: "Relações Afetivas, Familiares e Vida Cotidiana",
    5: "Entretenimento, Cultura e Produção Criativa",
    6: "Jogos e Esportes Competitivos",
    7: "Política, Religião e Disputas Ideológicas",
    8: "Ruído, Humor e Interações Genéricas",
    9: "Gênero, Sexualidade e Padrões de Atratividade",
    10: "Identidade, Raça e Relações Internacionais",
    11: "Trabalho, Economia e Condições Materiais",
    12: "Identidade, Raça e Relações Internacionais",
    13: "Entretenimento, Cultura e Produção Criativa",
    14: "Jogos e Esportes Competitivos",
    15: "Gênero, Sexualidade e Padrões de Atratividade",
    16: "Alimentação, Ética e Estilos de Vida",
    17: "Relações Afetivas, Familiares e Vida Cotidiana",
    18: "Corpo, Saúde e Apresentação Pessoal",
    19: "Política, Religião e Disputas Ideológicas",
    20: "Política, Religião e Disputas Ideológicas",
    21: "Corpo, Saúde e Apresentação Pessoal",
}

df_full["macro_topic"] = df_full["topic"].map(TOPIC_TO_MACROTOPIC)
df_full["macro_topic"].value_counts()
#%%
top_topics = ["Gênero, Sexualidade e Padrões de Atratividade","Relações Afetivas, Familiares e Vida Cotidiana"]

# 1. Cria a matriz de proporções uma única vez
topic_counts = pd.crosstab(
    df_full["subreddit"],
    df_full["macro_topic"],
    normalize="index"
)

# 2. Cria a coluna score_seed somando a proporção dos tópicos 4 e 9
topic_counts["score_seed"] = topic_counts[top_topics[0]] + topic_counts[top_topics[1]]

# 3. Gera o DataFrame de ranking já formatado e ordenado
ranking = (
    topic_counts[["score_seed"]]
    .sort_values(by="score_seed", ascending=False)
    .reset_index() # Transforma o index (subreddit) em coluna
)

print(ranking.head(50))
#%%
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(
    style="white",
    font="Liberation Sans"
)

plt.rcParams.update({
    'font.family': 'Liberation Sans',  
    'font.size': 22,
    'axes.titlesize': 22,
    'axes.labelsize': 22,
    'xtick.labelsize': 22,
    'ytick.labelsize': 22,
    'legend.fontsize': 22,
    'figure.titlesize': 22,
    'text.color': "#3F3F3FD8",
    'axes.labelcolor': "#3F3F3FD8",
    'xtick.color': "#3F3F3FD8",
    'ytick.color': "#3F3F3FD8",
    'font.weight': 700, 
    'axes.labelweight': 700,
    'axes.titleweight': 700,
})

format = 'png' 

# 1. Garante que o ranking está ordenado
ranking = ranking.sort_values(by="score_seed", ascending=False).reset_index(drop=True)

from scipy.stats import zscore

# ==========================================
# MÉTODO 1: Z-SCORE (Anomalia Estatística)
# ==========================================
ranking["z_score"] = zscore(ranking["score_seed"])

# Testando limiares acadêmicos comuns
z_2 = ranking[ranking["z_score"] >= 2.0]
z_3 = ranking[ranking["z_score"] >= 3.0]

print("\n===== RESULTADOS Z-SCORE =====")
print(f"Subreddits com Z >= 2.0 (Anomalias Fortes): {len(z_2)}")
print(f"Subreddits com Z >= 3.0 (Anomalias Extremas): {len(z_3)}")


# ==========================================
# VISUALIZAÇÃO DOS CORTES
# ==========================================
plt.figure(figsize=(10, 6))
#plt.plot(ranking.index, ranking["score_seed"], color='blue', linewidth=2, label='Score de Afinidade')

ranking["score_smooth"] = (
    ranking["score_seed"]
    .rolling(window=10, center=True)
    .mean()
)

plt.plot(
    ranking.index,
    ranking["score_smooth"],
    color='blue',
    linewidth=2.5
)

plt.fill_between(
    ranking.index,
    ranking["score_seed"],
    color='cornflowerblue',
    alpha=0.2
)

media = ranking["score_seed"].mean()

plt.axhline(
    y=media,
    color='green',
    linestyle='--',
    linewidth=2,
    label=f'Média ({media:.3f})'
)

std = ranking["score_seed"].std()

plt.axhspan(
    max(0, media - std),
    media + std,
    color='green',
    alpha=0.1,
    label='±1 DP'
)

cut = len(z_2)

plt.axvspan(
    0,
    cut,
    color='red',
    alpha=0.1,
    label='Subreddits selecionados'
)

plt.axhline(
    media + 2*std,
    color='orange',
    linestyle=':',
    linewidth=2,
    label='Limiar Z=2'
)

# Adiciona as linhas de corte no gráfico

plt.axvline(x=len(z_2)-1, color='red', linestyle='--', label=f'Z-Score >= 2 ({len(z_2)} subs)')

plt.xlim(left=0)
plt.ylim(bottom=0) 

plt.xlabel("Posição do Subreddit no Ranking")
plt.ylabel("Score de Afinidade")
plt.grid(True, which='major', linestyle='-', linewidth=0.75, alpha=0.55)
plt.minorticks_on()
plt.grid(True, which='minor', linestyle='-', linewidth=0.25, alpha=0.45)
plt.legend()
plt.tight_layout()

plt.savefig("zcore.png", dpi=300)

plt.show()

print(ranking["score_seed"].describe())

#%%
"""
ranking = ranking.reset_index()
ranking.columns = ["subreddit", "score_seed"]

corte = ranking["score_seed"].quantile(0.95)

nova_seed = ranking[
    ranking["score_seed"] >= corte
]
"""

nova_seed = ranking.loc[
    ranking["z_score"] >= 2,
    "subreddit"
].tolist()
#%%
#subreddits_misogin = df_ranking[df_ranking['n_keywords']>=31]['subreddit'].tolist()
#subreddits_misogin = nova_seed['subreddit'].tolist()
import json

dados_manosfera = {}
print(f"Quantidade de subreddits misóginos {len(nova_seed)}")
for sub in nova_seed:
    df_sub = df[df['subreddit'] == sub].copy()

    top_10_posts = (
        df_sub.sort_values(by="score", ascending=False)
        .head(10)
        .fillna("")
    )
    
    lista_posts_json = []
    for _, row in top_10_posts.iterrows():
        texto_bruto = str(row["text"]) if "text" in row else str(row.get("selftext", ""))
    
        # Se o texto for maior que 200, corta e põe '...', se não, deixa normal
        texto_cortado = texto_bruto[:200].strip() + "..." if len(texto_bruto) > 200 else texto_bruto

        post_dict = {
            "id": str(row["id"]),
            "text": texto_cortado
        }
        lista_posts_json.append(post_dict)
        
    # Salva no dicionário principal usando o nome do subreddit como chave
    dados_manosfera[sub] = {"top10_posts_score": lista_posts_json}

# 2. Salva o dicionário estruturado em um arquivo JSON limpo e identado
caminho_saida = Path(ROOT / "data" / "processed" / "top10_posts_83subreddits.json")
caminho_saida.parent.mkdir(parents=True, exist_ok=True)

with open(caminho_saida, "w", encoding="utf-8") as f:
    json.dump(dados_manosfera, f, indent=4, ensure_ascii=False)

print(f"Sucesso! Arquivo salvo em: {caminho_saida}")