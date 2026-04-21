# %%

import igraph as ig
import matplotlib.pyplot as plt
import os
import json
import pandas as pd
import itertools
from tqdm import tqdm
import math

# %%

rows = []
erros = []

# Itera por cada profundidade
for depth in range(3 + 1):
    dir_ = f"../data/raw/final/{depth}/subreddits"
    if not os.path.exists(dir_):
        continue
    for filename in os.listdir(dir_):
        if filename.endswith(".json"):
            path = os.path.join(dir_, filename)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    posts = json.load(f)
                # Adiciona a profundidade em cada post para rastreabilidade
                for post in posts:
                    post["depth"] = depth
                rows.extend(posts)
            except json.JSONDecodeError:
                erros.append(path)


df = pd.DataFrame(rows)

print(f"Arquivos com erro: {len(erros)}")
print(f"Quantidade de subreddits únicos: {df['subreddit'].nunique()}")
print(f"Quantidade de autores únicos: {df['author'].nunique()}")
print(f"Quantidade de posts carregados: {len(df)}")

# %%
df_0 = df[df['depth'] == 0]
df_1 = df[df['depth'] == 1]
df_2 = df[df['depth'] == 2]

#%%

#%%
# DataFrame com cada subreddit e seus autores únicos
subreddit_users = df.groupby('subreddit')['author'].unique().reset_index()

def get_subreddit_connections(df_grouped):
    # Cria um dicionário onde a chave é o subreddit e o valor é um conjunto de autores únicos
    sub_map = {row.subreddit: set(row.author) for row in df_grouped.itertuples()}
    
    subreddits = list(sub_map.keys())
    edge_list = []

    # Itera por todas as combinações de subreddits (sem repetição)
    # evitar comparar (A,B) e depois (B,A) desnecessariamente
    for sub_a, sub_b in itertools.combinations(subreddits, 2):

        common_authors = sub_map[sub_a].intersection(sub_map[sub_b])
        weight = len(common_authors)
        
        if weight > 0:
            # Se houver pelo menos um autor em comum, adiciona a conexão à lista de arestas
            edge_list.append({
                'source': sub_a,
                'target': sub_b,
                'weight': weight # número de autores em comum entre os dois subreddits
            })
    
    return pd.DataFrame(edge_list)

df_conexoes = get_subreddit_connections(subreddit_users)

# %%

g = ig.Graph.TupleList(
    df_conexoes.itertuples(index=False), 
    directed=False, 
    edge_attrs=['weight']
)

print(f"Quantidade de nós: {g.vcount()}")
print(f"Quantidade de arestas: {g.ecount()}")

# %%
g.write_graphml("../reports/grafo.graphml")
# %%
