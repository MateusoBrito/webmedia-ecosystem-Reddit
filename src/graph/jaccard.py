from pathlib import Path
import networkx as nx
import pandas as pd
from itertools import combinations
from src.utils.data_loader import load_preprocessed_data, ROOT

OUTPUT_PATH = Path(ROOT / "data/processed/graph_heuristic.graphml")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Limite mínimo de usuários compartilhados para criar uma aresta.
# Comece com 3. Se a rede ficar muito grande, suba para 5. Se der zero, desça para 2.
MIN_OVERLAP = 3 

def load_and_categorize(columns: list[str]) -> pd.DataFrame:
    # A MESMA FUNÇÃO QUE VOCÊ JÁ TEM, ESTÁ PERFEITA.
    df = load_preprocessed_data(columns)
    subreddits = pd.read_parquet(Path("src/graph/teste.parquet"))
    seed = subreddits['subreddit'].tolist()
    
    authors = df[(df['subreddit'].isin(seed)) & (df['author'] != '[deleted]')]['author'].tolist()
    autores_unicos = list(set(authors))
    
    atividades_dos_autores = df[df['author'].isin(autores_unicos)].copy()
    author_counts = atividades_dos_autores.groupby('author')['subreddit'].nunique()
    autores_conectores = author_counts[author_counts >= 3].index
    
    df_destilado = atividades_dos_autores[atividades_dos_autores['author'].isin(autores_conectores)]
    return df_destilado

def build_heuristic_graph(df, min_overlap=MIN_OVERLAP) -> nx.Graph:
    print(f"\nConstruindo a rede baseada em co-ocorrência (mínimo de {min_overlap} autores)...")
    
    # Dicionário: {subreddit: set(autores)}
    user_counts = df.groupby('subreddit')['author'].apply(set).to_dict()
    sub_names = list(user_counts.keys())
    
    G = nx.Graph()
    arestas_criadas = 0
    
    # Compara todos contra todos (Combinatória)
    for sub_a, sub_b in combinations(sub_names, 2):
        users_a = user_counts[sub_a]
        users_b = user_counts[sub_b]
        
        # Interseção (quantos postaram em ambos)
        overlap = len(users_a & users_b)
        
        if overlap >= min_overlap:
            # Jaccard (força da conexão)
            peso_jaccard = overlap / len(users_a | users_b)
            
            G.add_edge(sub_a, sub_b, weight=peso_jaccard, absolute_overlap=overlap)
            arestas_criadas += 1

    print(f"Arestas criadas (Overlaps >= {min_overlap}): {arestas_criadas}")

    # Limpar subreddits isolados
    isolados = [n for n in list(G.nodes()) if G.degree(n) == 0]
    G.remove_nodes_from(isolados)
    
    print(f"Grafo Final: {G.number_of_nodes()} nós, {G.number_of_edges()} arestas")
    return G

if __name__ == "__main__":
    df = load_and_categorize(["author", "subreddit"])
    G = build_heuristic_graph(df, min_overlap=MIN_OVERLAP)
    
    if G.number_of_edges() > 0:
        nx.write_graphml(G, OUTPUT_PATH)
        print(f"\nGrafo salvo com sucesso em: {OUTPUT_PATH}")
    else:
        print("\nO grafo está vazio. Tente diminuir o MIN_OVERLAP.")