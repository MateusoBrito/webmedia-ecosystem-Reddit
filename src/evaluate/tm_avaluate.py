#%%
import sys
import os

sys.path.append('/home/mateus/reddit')
from src.utils.data_loader import load_preprocessed_data,load_ranking_subreddits,load_toxicity,ROOT
import pandas as pd
from pathlib import Path

df = load_preprocessed_data()
df_topic = pd.read_parquet(Path(ROOT/"reports"/"topic_modeling5"/"best_combined_sdc_rank2"/"post_topics.parquet"))
df_toxicity = load_toxicity()
df_ranking = load_ranking_subreddits()
# %%
print(f"Colunas do df original: {df.columns}")
print(f"Colunas do df dos topicos: {df_topic.columns}")
print(f"Colunas do df da toxicidade: {df_toxicity.columns}")
print(f"Colunas do df do ranking: {df_ranking.columns}")
#%%
print(f"Tamanho do df original: {len(df)}")
print(f"Tamanho do df dos topicos: {len(df_topic)}")
print(f"Tamanho do df da toxicidade: {len(df_toxicity)}")
print(f"Tamanho do df do ranking: {len(df_ranking)}")
#%%
df = df[df['depth'] == 0]
#%%
df_full = pd.merge(df,df_topic,on="id",how="inner")
df_full = pd.merge(df_full, df_toxicity,on="id",how="inner")
df_full = pd.merge(df_full, df_ranking, on="subreddit",how="inner")

print(f"Colunas do df resultante: {df_full.columns}")
print(f"Tamanho do df resultante: {len(df_full)}")
#%%
df_clean = df_full.dropna()
print(f"Linhas após dropna: {len(df_clean)}")
# %%
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np

#%%
subreddit_agg = df_clean.groupby('n_keywords').agg(
    tox_media=('perspective_toxicity', 'mean'),
    n_posts=('n_keywords', 'count')
).reset_index()

corr_agg, pval_agg = stats.spearmanr(subreddit_agg['n_keywords'], subreddit_agg['tox_media'])
print(f"Spearman agregado: r = {corr_agg:.4f}, p = {pval_agg:.4f}")
#%%
fig, ax = plt.subplots(figsize=(8, 5))

ax.scatter(subreddit_agg['n_keywords'], subreddit_agg['tox_media'], 
           s=subreddit_agg['n_posts']/50, alpha=0.6, color='steelblue')

# Linha de tendência
m, b = np.polyfit(subreddit_agg['n_keywords'], subreddit_agg['tox_media'], 1)
x_line = np.linspace(subreddit_agg['n_keywords'].min(), subreddit_agg['n_keywords'].max(), 100)
ax.plot(x_line, m * x_line + b, color='red', linestyle='--')

ax.set_xlabel('n_keywords (imersão na manosphere)')
ax.set_ylabel('Toxicidade média (Perspective API)')
ax.set_title(f'Toxicidade média por imersão na manosphere\nSpearman r={corr_agg:.3f}, p={pval_agg:.4f}')
ax.annotate('Tamanho do ponto ∝ número de posts', xy=(0.02, 0.95), 
            xycoords='axes fraction', fontsize=9, color='gray')

plt.tight_layout()
plt.savefig('agregado_toxicity_keywords.png', dpi=150)
plt.show()
#%%
# Agrega por subreddit
subreddit_agg = df_clean.groupby(['subreddit', 'n_keywords']).agg(
    tox_mean=('perspective_toxicity', 'mean'),
    tox_median=('perspective_toxicity', 'median'),
    severe_toxicity=('severe_toxicity', 'mean'),
    identity_attack=('identity_attack', 'mean'),
    insult=('insult', 'mean'),
    profanity=('profanity', 'mean'),
    threat=('threat', 'mean'),
    n_posts=('perspective_toxicity', 'count')
).reset_index()

# Score composto: média das 6 dimensões
toxicity_cols = ['tox_mean', 'severe_toxicity', 'identity_attack', 'insult', 'profanity', 'threat']
subreddit_agg['tox_score'] = subreddit_agg[toxicity_cols].mean(axis=1)

print(subreddit_agg.sort_values('tox_score', ascending=False).head(10))
print(f"\nTotal subreddits: {len(subreddit_agg)}")
#%%
# Spearman para cada dimensão de toxicidade
for col in toxicity_cols + ['tox_score']:
    col_name = col.replace('tox_mean', 'perspective_toxicity')
    r, p = stats.spearmanr(subreddit_agg['n_keywords'], subreddit_agg[col])
    print(f"{col:<25} r={r:.4f}  p={p:.4f}")
#%%
fig, ax = plt.subplots(figsize=(8, 5))

labels = ['tox_mean', 'severe_toxicity', 'identity_attack', 'insult', 'profanity', 'threat', 'tox_score']
r_values = [0.1313, 0.1034, 0.1892, 0.1230, 0.1070, 0.1333, 0.1321]
colors = ['steelblue' if l != 'identity_attack' else 'crimson' for l in labels]

bars = ax.barh(labels, r_values, color=colors)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel('Spearman r (correlação com n_keywords)')
ax.set_title('Correlação entre imersão na manosphere\ne dimensões de toxicidade (nível subreddit)')

for bar, val in zip(bars, r_values):
    ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('correlacao_dimensoes_toxicidade.png', dpi=150)
plt.show()

#%%
subreddit_agg.sort_values('tox_score')
#%%
# Distribuição de topics por faixa de n_keywords
df_clean['kw_faixa'] = pd.cut(df_clean['n_keywords'], 
                                    bins=[0, 15, 30, 45, 60], 
                                    labels=['baixo (1-15)', 'médio (16-30)', 'alto (31-45)', 'muito alto (46-60)'])

print(df_clean['topic'].value_counts())
print(df_clean['kw_faixa'].value_counts())

#%%
# Distribuição de topics por faixa de n_keywords
df_clean['kw_faixa'] = pd.cut(df_clean['n_keywords'], 
                              bins=[0, 20, 40, 60], 
                              labels=['baixo (1-20)', 'médio (21-40)', 'alto (41-60)'])

print(df_clean['topic'].value_counts())
print(df_clean['kw_faixa'].value_counts())

#%%
df_clean_full = df_clean.copy() 
df_clean_full['kw_faixa'] = pd.cut(df_clean_full['n_keywords'],
                                   bins=[0, 20, 40, 60],
                                   labels=['baixo (1-20)', 'médio (21-40)', 'alto (41-60)'])

# ── 1. PROPORÇÃO DE TOPICS POR FAIXA ─────────────────────────────────────────
topic_faixa = (
    df_clean_full.groupby(['kw_faixa', 'topic'], observed=True)
    .size()
    .reset_index(name='count')
)
pivot = topic_faixa.pivot(index='topic', columns='kw_faixa', values='count').fillna(0)
pivot_norm = pivot.div(pivot.sum(axis=0), axis=1)  # proporção por faixa

# ── 2. TOXICIDADE MÉDIA POR TOPIC E FAIXA ────────────────────────────────────
tox_topic_faixa = (
    df_clean_full.groupby(['kw_faixa', 'topic'], observed=True)['perspective_toxicity']
    .mean()
    .reset_index()
    .pivot(index='topic', columns='kw_faixa', values='perspective_toxicity')
)

# ── PLOTS ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.heatmap(pivot_norm, ax=axes[0], cmap='YlOrRd', annot=True, fmt='.2f',
            linewidths=0.3, cbar_kws={'label': 'proporção'})
axes[0].set_title('Proporção de cada topic por faixa de n_keywords')
axes[0].set_xlabel('Faixa de keywords')
axes[0].set_ylabel('Topic')

sns.heatmap(tox_topic_faixa, ax=axes[1], cmap='YlOrRd', annot=True, fmt='.3f',
            linewidths=0.3, cbar_kws={'label': 'toxicidade média'})
axes[1].set_title('Toxicidade média por topic e faixa de n_keywords')
axes[1].set_xlabel('Faixa de keywords')
axes[1].set_ylabel('Topic')

plt.tight_layout()
plt.savefig('topic_keywords_toxicidade.png', dpi=150, bbox_inches='tight')
plt.show()

# ── RESUMO: qual topic cresce mais nas faixas altas? ─────────────────────────
# ATUALIZADO: Calculando a variação entre a nova faixa alta (41-60) e a baixa (1-20)
pivot_norm['variacao'] = pivot_norm['alto (41-60)'] - pivot_norm['baixo (1-20)']
print("\nTopics que mais crescem nas faixas altas de keywords:")
print(pivot_norm['variacao'].sort_values(ascending=False))
#%%