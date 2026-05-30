"""
SDSM — Stochastic Degree Sequence Model
Backbone extraction for bipartite projections.

Referência: Neal, Z. P. (2014). The backbone of bipartite projections.
            Social Networks, 39, 84–97.
            Neal, Domagalski & Sagan (2021). Comparing alternatives to FDSM.
            Scientific Reports, 11, 23929.

Ideia central
─────────────
Dado um grafo bipartido B (usuários × subreddits), queremos saber:
"Dois subreddits A e B compartilham usuários ALÉM do que o acaso explicaria?"

O SDSM responde assim:
  1. Para cada par (A, B), calcula a probabilidade p_ik de um usuário i
     estar conectado ao subreddit k no mundo aleatório — preservando as
     proporções de atividade de cada usuário e tamanho de cada subreddit.
  2. Com essas probabilidades, determina a distribuição esperada do peso
     da aresta (A,B) = número de usuários em comum.
  3. Compara o peso observado com essa distribuição → p-value.
  4. Aresta entra no backbone se o p-value < alpha (com correção FDR).
"""

import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import poisson
from scipy.special import comb
from pathlib import Path
from itertools import combinations
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# 1. Construção da matriz bipartida
# ─────────────────────────────────────────────────────────────────────────────

def build_biadjacency(df: pd.DataFrame,
                      user_col: str = "author",
                      item_col: str = "subreddit") -> tuple:
    """
    Transforma o DataFrame em uma matriz bipartida binária B.

    B[i, k] = 1  se o usuário i publicou no subreddit k
    B[i, k] = 0  caso contrário

    Retorna
    -------
    B         : np.ndarray  shape (n_users, n_subs)
    user_list : list de nomes dos usuários (índice das linhas)
    sub_list  : list de nomes dos subreddits (índice das colunas)
    """
    user_list = sorted(df[user_col].unique())
    sub_list  = sorted(df[item_col].unique())

    user_idx = {u: i for i, u in enumerate(user_list)}
    sub_idx  = {s: j for j, s in enumerate(sub_list)}

    n_users = len(user_list)
    n_subs  = len(sub_list)

    B = np.zeros((n_users, n_subs), dtype=np.float32)
    for _, row in df.iterrows():
        i = user_idx[row[user_col]]
        j = sub_idx[row[item_col]]
        B[i, j] = 1.0

    print(f"Matriz bipartida: {n_users} usuários × {n_subs} subreddits")
    print(f"Arestas: {int(B.sum()):,}  |  Densidade: {B.mean():.4f}")
    return B, user_list, sub_list


# ─────────────────────────────────────────────────────────────────────────────
# 2. Estimativa das probabilidades de célula (coração do SDSM)
# ─────────────────────────────────────────────────────────────────────────────

def estimate_probabilities(B: np.ndarray,
                           max_iter: int = 25,
                           tol: float = 1e-6) -> np.ndarray:
    """
    Estima P[i,k] = probabilidade de o usuário i estar conectado ao subreddit k
    no mundo aleatório, preservando os graus esperados de linhas e colunas.

    Algoritmo: IPF — Iterative Proportional Fitting (alias RAS ou Sinkhorn)
    ─────────────────────────────────────────────────────────────────────────
    Queremos uma matriz de probabilidades P tal que:
      • Soma das linhas  ≈ graus observados dos usuários  (row_sums)
      • Soma das colunas ≈ graus observados dos subreddits (col_sums)

    Começamos com P = outer(row_prob, col_prob) e ajustamos alternadamente:
      Passo A: normaliza linhas  → P[i,:] *= row_sums[i] / P[i,:].sum()
      Passo B: normaliza colunas → P[:,k] *= col_sums[k] / P[:,k].sum()

    Isso converge para a matriz de máxima entropia com as margens desejadas.
    É exatamente o BiCM no limite contínuo.

    Parâmetros
    ----------
    B        : matriz bipartida binária (n_users × n_subs)
    max_iter : número máximo de iterações IPF
    tol      : critério de convergência (erro nas margens)

    Retorna
    -------
    P : np.ndarray  shape (n_users, n_subs) — probabilidades de célula
    """
    row_sums = B.sum(axis=1)   # grau de cada usuário
    col_sums = B.sum(axis=0)   # grau de cada subreddit
    total    = B.sum()

    # Inicialização: produto externo das proporções marginais
    P = np.outer(row_sums / total, col_sums / total) * total
    # Garante que P fique em (0,1) — necessário para o teste de Poisson depois
    P = np.clip(P, 1e-10, 1 - 1e-10)

    for iteration in range(max_iter):
        P_old = P.copy()

        # Normaliza linhas
        row_actual = P.sum(axis=1)
        row_scale  = np.where(row_actual > 0, row_sums / row_actual, 0)
        P *= row_scale[:, np.newaxis]
        P  = np.clip(P, 1e-10, 1 - 1e-10)

        # Normaliza colunas
        col_actual = P.sum(axis=0)
        col_scale  = np.where(col_actual > 0, col_sums / col_actual, 0)
        P *= col_scale[np.newaxis, :]
        P  = np.clip(P, 1e-10, 1 - 1e-10)

        # Critério de parada
        err = np.abs(P - P_old).max()
        if err < tol:
            print(f"IPF convergiu em {iteration+1} iterações (erro={err:.2e})")
            break
    else:
        print(f"IPF atingiu max_iter={max_iter} (erro={err:.2e})")

    return P.astype(np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Cálculo dos p-values via distribuição de Poisson
# ─────────────────────────────────────────────────────────────────────────────

def compute_pvalue_pair(p_i: np.ndarray, p_j: np.ndarray,
                        observed: int) -> float:
    """
    Calcula o p-value para o par de subreddits (i, j).

    Para cada usuário u, a probabilidade de ele estar em AMBOS os subreddits
    i e j no mundo aleatório é:
        q_u = P[u, i] * P[u, j]

    O número total de usuários em comum é a soma de variáveis de Bernoulli
    independentes com parâmetros q_u. Aproximamos essa soma por uma Poisson
    com λ = Σ q_u (Approximation de Le Cam — válida quando q_u são pequenos
    e independentes).

    p-value = P(X ≥ observed)  onde  X ~ Poisson(λ)
    """
    lam = np.dot(p_i, p_j)          # λ = soma das probabilidades de co-ocorrência
    if lam <= 0:
        return 1.0
    # P(X >= obs) = 1 - P(X <= obs-1) = survival function em obs-1
    pval = poisson.sf(observed - 1, lam)
    return float(pval)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Correção de múltiplos testes — FDR (Benjamini-Hochberg)
# ─────────────────────────────────────────────────────────────────────────────

def fdr_correction(pvalues: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """
    Benjamini-Hochberg FDR correction.

    Para m testes, ordena os p-values e aplica:
        threshold_k = (k/m) * alpha
    Rejeita as hipóteses nulas para os k* menores p-values onde
    p_(k) <= threshold_k.

    Retorna máscara booleana: True = aresta significativa (entra no backbone).
    """
    m = len(pvalues)
    if m == 0:
        return np.array([], dtype=bool)

    order   = np.argsort(pvalues)
    ranked  = np.empty(m, dtype=int)
    ranked[order] = np.arange(1, m + 1)

    thresholds = (ranked / m) * alpha
    reject = pvalues <= thresholds

    # BH: rejeitar todas abaixo do maior k* que passa
    if reject.any():
        max_k = ranked[reject].max()
        reject = ranked <= max_k

    return reject


# ─────────────────────────────────────────────────────────────────────────────
# 5. Pipeline principal
# ─────────────────────────────────────────────────────────────────────────────

def run_sdsm(B: np.ndarray,
             sub_list: list,
             alpha: float = 0.05,
             min_observed: int = 1) -> nx.Graph:
    """
    Executa o SDSM completo e retorna o grafo backbone.

    Parâmetros
    ----------
    B            : matriz bipartida (n_users × n_subs)
    sub_list     : nomes dos subreddits (colunas de B)
    alpha        : nível de significância após correção FDR
    min_observed : filtra pares com menos de N usuários em comum antes
                   do teste (reduz custo computacional sem perder sinal)

    Retorna
    -------
    G : grafo NetworkX com arestas validadas
        atributo 'weight'  = usuários em comum
        atributo 'pvalue'  = p-value após FDR
        atributo 'lambda'  = λ esperado pelo modelo nulo
    """
    n_subs = B.shape[1]

    # ── 2. Estima probabilidades de célula
    print("\nEstimando probabilidades de célula (IPF)...")
    P = estimate_probabilities(B)
    # P[:, k] = probabilidade de cada usuário estar no subreddit k no nulo
    # Transposta para indexação por subreddit: P_cols[k, :] = vetor do sub k
    P_cols = P.T   # shape: (n_subs, n_users)

    # ── 3. Projeção observada: P_obs[i,j] = usuários em comum entre subs i e j
    print("Calculando projeção observada (B^T × B)...")
    P_obs = (B.T @ B).astype(int)   # shape: (n_subs, n_subs)

    # ── 4. Calcula p-values para todos os pares
    pairs     = list(combinations(range(n_subs), 2))
    pvalues   = np.ones(len(pairs))
    observed  = np.zeros(len(pairs), dtype=int)
    lambdas   = np.zeros(len(pairs))

    print(f"Calculando p-values para {len(pairs):,} pares de subreddits...")
    for idx, (i, j) in enumerate(tqdm(pairs, desc="SDSM p-values")):
        obs = P_obs[i, j]
        observed[idx] = obs
        if obs < min_observed:
            pvalues[idx] = 1.0
            continue
        lam = float(np.dot(P_cols[i], P_cols[j]))
        lambdas[idx] = lam
        pvalues[idx] = poisson.sf(obs - 1, lam) if lam > 0 else 1.0

    # ── 5. Correção FDR
    print("Aplicando correção FDR (Benjamini-Hochberg)...")
    significant = fdr_correction(pvalues, alpha=alpha)

    n_sig = significant.sum()
    print(f"Arestas significativas (α={alpha}): {n_sig:,} / {len(pairs):,}")

    # ── 6. Constrói o grafo
    G = nx.Graph()
    G.add_nodes_from(sub_list)

    for idx, (i, j) in enumerate(pairs):
        if significant[idx]:
            G.add_edge(
                sub_list[i], sub_list[j],
                weight=int(observed[idx]),
                pvalue=float(pvalues[idx]),
                lam=float(lambdas[idx])
            )

    # Remove isolados
    isolados = [n for n in G.nodes() if G.degree(n) == 0]
    G.remove_nodes_from(isolados)

    print(f"\nGrafo backbone: {G.number_of_nodes()} nós, {G.number_of_edges()} arestas")
    return G


# ─────────────────────────────────────────────────────────────────────────────
# 6. Integração com seu pipeline existente
# ─────────────────────────────────────────────────────────────────────────────

def build_sdsm_graph(df: pd.DataFrame,
                     alpha: float = 0.05,
                     min_observed: int = 2,
                     output_path: Path = None) -> nx.Graph:
    """
    Wrapper completo: DataFrame → grafo backbone via SDSM.

    Parâmetros
    ----------
    df           : DataFrame com colunas 'author' e 'subreddit'
    alpha        : nível de significância FDR
    min_observed : mínimo de usuários em comum para testar o par
    output_path  : se fornecido, salva o grafo em .graphml

    Retorna
    -------
    G : grafo NetworkX validado
    """
    # Constrói matriz bipartida
    B, user_list, sub_list = build_biadjacency(df)

    # Roda SDSM
    G = run_sdsm(B, sub_list, alpha=alpha, min_observed=min_observed)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        nx.write_graphml(G, output_path)
        print(f"Grafo salvo em: {output_path}")

    return G


# ─────────────────────────────────────────────────────────────────────────────
# 7. Teste rápido com dados sintéticos
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import random

    print("=" * 60)
    print("TESTE COM DADOS SINTÉTICOS")
    print("=" * 60)

    # Cria 3 clusters bem definidos de subreddits
    # Cluster 1: subs 0-4  → usuários 0-49   (muito overlap interno)
    # Cluster 2: subs 5-9  → usuários 50-99
    # Cluster 3: subs 10-14→ usuários 100-149
    # Noise: alguns usuários cruzam clusters (sinal fraco)

    random.seed(42)
    np.random.seed(42)

    n_users_per_cluster = 50
    n_subs_per_cluster  = 5
    n_clusters          = 3
    noise_prob          = 0.05  # probabilidade de cruzar clusters

    records = []
    for c in range(n_clusters):
        users_base = range(c * n_users_per_cluster, (c+1) * n_users_per_cluster)
        subs_base  = [f"sub_{c*n_subs_per_cluster + s}" for s in range(n_subs_per_cluster)]
        all_subs   = [f"sub_{i}" for i in range(n_clusters * n_subs_per_cluster)]

        for u in users_base:
            # Publica em 2-4 subs do próprio cluster
            n_own = random.randint(2, 4)
            for s in random.sample(subs_base, min(n_own, len(subs_base))):
                records.append({"author": f"u{u}", "subreddit": s})
            # Talvez publique em um sub de outro cluster (ruído)
            if random.random() < noise_prob:
                other_subs = [s for s in all_subs if s not in subs_base]
                records.append({"author": f"u{u}", "subreddit": random.choice(other_subs)})

    df_test = pd.DataFrame(records).drop_duplicates()
    print(f"\nDataset sintético: {df_test['author'].nunique()} usuários, "
          f"{df_test['subreddit'].nunique()} subreddits, "
          f"{len(df_test)} registros")

    G = build_sdsm_graph(df_test, alpha=0.05, min_observed=2)

    print("\nArestas encontradas:")
    for u, v, d in sorted(G.edges(data=True), key=lambda x: x[2]['pvalue']):
        print(f"  {u} — {v}  | shared={d['weight']}  λ={d['lam']:.2f}  p={d['pvalue']:.4f}")