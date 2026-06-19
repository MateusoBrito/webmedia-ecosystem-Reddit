#%%
import sys
import os

sys.path.append('/home/mateus/reddit')
from src.utils.data_loader import load_preprocessed_data,load_ranking_subreddits,load_toxicity,ROOT
import pandas as pd
from pathlib import Path

df = load_preprocessed_data()
df_topic = pd.read_parquet(Path(ROOT/"data"/"processed"/"post_topics_1.parquet"))
df_full = pd.merge(df,df_topic,on="id",how="inner")

subreddits_pro = ["askmrp", "blackpillscience", "marriedredpill", "redpillwomen", "seduction", "semenretention", "theredpill"]
subreddits_contra = ["askfeminists", "askgaybros", "askgaybrosover30", "asktransgender", "exredpill", "incelexit", "letgirlshavefun", "menslib", "mtf", "mypartneristrans", "straighttransgirls"]

#%%
# 1. Isola os autores válidos da semente (excluindo os deletados)
authors_pro = df_full[
    (df_full['subreddit'].isin(subreddits_pro)) & 
    (df_full['author'] != '[deleted]')
]['author'].tolist()

autores_unicos = list(set(authors_pro))
print(f"Quantidade de autores na semente pro: {len(autores_unicos):,}")

authors_contra = df_full[
    (df_full['subreddit'].isin(subreddits_contra)) & 
    (df_full['author'] != '[deleted]')
]['author'].tolist()
autores_unicos_contra = list(set(authors_contra))
print(f"Quantidade de autores na semente contra: {len(autores_unicos_contra):,}")
#%%

subreddits_expandido = df_full[df_full['author'].isin(autores_unicos)]['subreddit'].dropna().unique().tolist()
print(f"Quantidade de subreddits na rede expandida (antes de filtrar a Diáspora): {len(subreddits_expandido):,}")
subreddits_novos = set(subreddits_expandido) - set(subreddits_pro)
print(f"Quantidade de subreddits novos (antes de filtrar a Diáspora): {len(subreddits_novos):,}")

subreddits_expandido_contra = df_full[df_full['author'].isin(autores_unicos_contra)]['subreddit'].dropna().unique().tolist()
print(f"Quantidade de subreddits na rede expandida (antes de filtrar a Diáspora): {len(subreddits_expandido_contra):,}")
subreddits_novos_contra = set(subreddits_expandido_contra) - set(subreddits_contra)
print(f"Quantidade de subreddits novos (antes de filtrar a Diáspora): {len(subreddits_novos_contra):,}")

#%%
"""
# 4. Salva o DATAFRAME COMPLETO (com os textos) para a Modelagem de Tópicos
#caminho_diaspora = Path(ROOT / "data" / "processed" / "posts_diaspora.parquet")
#caminho_diaspora.parent.mkdir(parents=True, exist_ok=True)

# Selecionamos apenas as colunas úteis para não deixar o arquivo pesado
#colunas_para_salvar = ["id", "author", "subreddit", "text", "text_clean", "depth"]
#df_diaspora[colunas_para_salvar].to_parquet(caminho_diaspora, index=False)
#print(f"Dataset da Diáspora salvo com sucesso em: {caminho_diaspora}")
"""
#%%
# 5. Salva também apenas a listagem de subreddits novos 
# (Útil caso precise rodar estatísticas descritivas ou o BiCM depois)
df_saida_subs = pd.DataFrame({'subreddit': list(subreddits_novos)})
caminho_subs = Path(ROOT / "data" / "processed" / "subreddits_pro_expanded_network.parquet")
df_saida_subs.to_parquet(caminho_subs, index=False)
print(f"Lista de subreddits expandidos salva em: {caminho_subs}")

df_saida_subs = pd.DataFrame({'subreddit': list(subreddits_novos_contra)})
caminho_subs = Path(ROOT / "data" / "processed" / "subreddits_contra_expanded_network.parquet")
df_saida_subs.to_parquet(caminho_subs, index=False)
print(f"Lista de subreddits expandidos salva em: {caminho_subs}")

#%%
# 6. Avaliação de perdas pelo rótulo [deleted]
posts_alvo = df_full[df_full['subreddit'].isin(subreddits_pro)]
total_alvo = len(posts_alvo)

posts_deleted = posts_alvo[posts_alvo['author'] == '[deleted]']
total_deleted = len(posts_deleted)

if total_alvo > 0:
    porcentagem = (total_deleted / total_alvo) * 100
    print(f"\nTotal de posts na bolha misógina: {total_alvo:,}")
    print(f"Posts feitos por [deleted]: {total_deleted:,} ({porcentagem:.2f}%)")
else:
    print("\nNenhum post encontrado para essa regra.")
# %%
