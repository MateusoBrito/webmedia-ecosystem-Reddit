import networkx as nx
import pandas as pd
import numpy as np
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests
from src.utils.data_loader import load_raw_data, ROOT

def build_validated_projection(df, alpha=0.05, min_shared=2):
    user_sets  = df.groupby('subreddit')['author'].apply(set).to_dict()
    # grau de cada usuário = quantos subreddits ele frequenta
    user_degree = df.groupby('author')['subreddit'].nunique().to_dict()
    
    subs   = list(user_sets.keys())
    N_subs = len(subs)  # número total de subreddits = população do teste

    edges = []
    for i in range(len(subs)):
        for j in range(i + 1, len(subs)):
            a, b   = subs[i], subs[j]
            set_a  = user_sets[a]
            set_b  = user_sets[b]
            shared = len(set_a & set_b)

            if shared < min_shared:
                continue

            size_a  = len(set_a)
            size_b  = len(set_b)
            jaccard = shared / (size_a + size_b - shared)

            # Hipergeométrica conforme Neal (2022):
            # N = total de subreddits
            # K = tamanho do subreddit A (quantos usuários ele tem)
            # n = tamanho do subreddit B
            # k = usuários em comum
            pval = hypergeom.sf(shared - 1, N_subs, size_a, size_b)

            edges.append((a, b, jaccard, pval))

    if not edges:
        print("Nenhum par com usuários suficientes em comum.")
        return nx.Graph()

    pvalues = np.array([e[3] for e in edges])
    _, pvals_corrected, _, _ = multipletests(pvalues, alpha=alpha, method='fdr_bh')

    G = nx.Graph()
    for (a, b, jaccard, _), pval_corr in zip(edges, pvals_corrected):
        if pval_corr < alpha:
            G.add_edge(a, b, weight=jaccard, pvalue=pval_corr)

    isolados = [n for n in G.nodes() if G.degree(n) == 0]
    G.remove_nodes_from(isolados)

    print(f"Pares testados:    {len(edges):,}")
    print(f"Arestas validadas: {G.number_of_edges():,}  (α={alpha}, FDR)")
    print(f"Grafo final:       {G.number_of_nodes()} nós, {G.number_of_edges()} arestas")
    return G

def load_and_categorize(columns: list[str]) -> pd.DataFrame:
    """Carrega os dados e fixa a ordem das categorias."""
    df = load_raw_data(columns)
    for col in columns:
        df[col] = pd.Categorical(df[col])
    return df

df = load_and_categorize(["author", "subreddit"])  
G = build_validated_projection(df, alpha=0.05, min_shared=2)

import community as community_louvain  # pip install python-louvain
import networkx as nx

# Detecta comunidades
partition = community_louvain.best_partition(G, weight='weight', random_state=42)

# Adiciona como atributo dos nós
nx.set_node_attributes(G, partition, 'community')

# Resumo
from collections import Counter
community_sizes = Counter(partition.values())
print(f"Comunidades detectadas: {len(community_sizes)}")
for comm_id, size in sorted(community_sizes.items(), key=lambda x: -x[1]):
    print(f"  Comunidade {comm_id}: {size} subreddits")

# Modularidade
communities = [{n for n, c in partition.items() if c == comm}
               for comm in set(partition.values())]
Q = nx.community.modularity(G, communities)
print(f"\nModularidade: {Q:.3f}")