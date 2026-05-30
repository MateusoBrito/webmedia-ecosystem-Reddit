#%%
from pathlib import Path
import networkx as nx
import os
import sys
import pandas as pd
import numpy as np
import json
from bicm import BipartiteGraph
from src.utils.data_loader import load_raw_data

# =========================
# FUNÇÕES
# =========================

def load_and_categorize(columns: list[str]) -> pd.DataFrame:
    """Carrega os dados e fixa a ordem das categorias."""
    df = load_raw_data(columns)
    for col in columns:
        df[col] = pd.Categorical(df[col])
    return df


def binarize(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Remove duplicatas para binarizar a rede."""
    return df.drop_duplicates(subset=columns)


def build_edgelist(df: pd.DataFrame, row_col: str, col_col: str) -> list[tuple]:
    """Gera edgelist com códigos inteiros consistentes com as categorias."""
    row = df[row_col].cat.codes.to_numpy()
    col = df[col_col].cat.codes.to_numpy()
    return list(zip(row, col))


def get_category_maps(df: pd.DataFrame, columns: list[str]) -> dict[str, dict]:
    """Retorna dicionários {índice: categoria} para cada coluna categórica."""
    return {
        col: dict(enumerate(df[col].cat.categories))
        for col in columns
    }


def run_bicm(edgelist: list[tuple], alpha: float = 0.01) -> np.ndarray:
    #max_threads = os.cpu_count()
    #print(f"Usando {max_threads} threads para o cálculo...")

    """Ajusta o BiCM e retorna a matriz de projeção das colunas."""
    bipartite_net = BipartiteGraph(edgelist=edgelist)
    bipartite_net.compute_projection(
        rows=False,
        alpha=alpha,
        approx_method="normal",
        validation_method="fdr",
        threads_num=4,
        progress_bar=True,
    )
    return bipartite_net.get_cols_projection()


def validate_projection(W: np.ndarray, expected_size: int) -> None:
    """Verifica se a matriz de projeção tem a dimensão esperada."""
    assert W.shape == (expected_size, expected_size), (
        f"ERRO: dimensões inconsistentes! "
        f"Esperado ({expected_size}, {expected_size}), obtido {W.shape}"
    )
    print(f"✓ Dimensões consistentes ({W.shape}) — mapeamento de índices garantido.")


def build_graph(W: np.ndarray, node_map: dict) -> nx.Graph:
    """Constrói o grafo NetworkX a partir da matriz, renomeia nós e remove isolados."""
    G = nx.from_numpy_array(W)
    G = nx.relabel_nodes(G, node_map)
    G.remove_nodes_from(list(nx.isolates(G)))
    return G


def save_outputs(
    G: nx.Graph,
    node_map: dict,
    graph_path: Path = Path("artifacts") / "graph" / "rede_bicm_fdr.graphml",
    map_path: Path = Path("artifacts") / "graph" / "subreddit_map.json",
) -> None:
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    map_path.parent.mkdir(parents=True, exist_ok=True)

    nx.write_graphml(G, graph_path)
    print(f"Rede salva em: {graph_path}")

    with open(map_path, "w") as f:
        json.dump({str(k): v for k, v in node_map.items()}, f, indent=2)
    print(f"Mapeamento salvo em: {map_path}")


# =========================
# PIPELINE PRINCIPAL
# =========================

def main():
    # 1. Carregar e categorizar
    df = load_and_categorize(["author", "subreddit"])

    author_counts = df.groupby('author')['subreddit'].nunique()

    # Manter apenas autores que postaram em pelo menos 2 subreddits (conectores de rede)
    autores_conectores = author_counts[author_counts >= 2].index
    df = df[df['author'].isin(autores_conectores)]

    print(f"Reduzimos o DF para focar apenas nos autores que criam pontes!")

    # 2. Mapas de índices (fixados antes da binarização)
    maps = get_category_maps(df, ["author", "subreddit"])
    author_map = maps["author"]
    subreddit_map = maps["subreddit"]

    print(f"Autores únicos:    {len(author_map)}")
    print(f"Subreddits únicos: {len(subreddit_map)}")

    # 3. Binarizar e gerar edgelist
    df = binarize(df, ["author", "subreddit"])
    edgelist = build_edgelist(df, row_col="author", col_col="subreddit")

    # 4. BiCM
    W = run_bicm(edgelist, alpha=0.01)

    # 5. Validar
    validate_projection(W, expected_size=len(subreddit_map))

    # 6. Grafo
    G = build_graph(W, subreddit_map)
    print(f"Nós: {G.number_of_nodes()}, Arestas: {G.number_of_edges()}")

    # 7. Salvar
    save_outputs(G, subreddit_map)


if __name__ == "__main__":
    main()