import pandas as pd
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from src.utils.data_loader import load_centroid_data

def build_semantic_graph(df, subreddit_col='subreddit', threshold=0.75, k_neighbors=None):
    """
    Calcula a similaridade de cosseno entre os centroides e reconstrói o grafo.
    
    Parâmetros:
    - df: DataFrame carregado pelo 'load_centroid_data'.
    - subreddit_col: Nome da coluna que identifica o subreddit (se for o índice, ajuste o código).
    - threshold: Valor mínimo de similaridade de cosseno (0 a 1) para criar uma aresta.
    - k_neighbors: Se definido (ex: 5), força cada nó a se conectar apenas aos X nós mais parecidos.
    """
    # Configurar o identificador como índice se já não for
    if subreddit_col in df.columns:
        df = df.set_index(subreddit_col)
    
    # 1. Calcular a matriz de similaridade de cosseno (Matrix: N x N)
    print("Calculando a matriz de similaridade de cosseno...")
    sim_matrix = cosine_similarity(df.values)
    
    # Converter para DataFrame para manter os rótulos legíveis
    sim_df = pd.DataFrame(sim_matrix, index=df.index, columns=df.index)
    
    # 2. Inicializar o grafo do NetworkX
    G = nx.Graph()
    G.add_nodes_from(df.index)
    
    subreddits = list(df.index)
    total_subs = len(subreddits)
    
    # 3. Estratégia A: Filtragem por Limiar Estrito (Threshold)
    if k_neighbors is None:
        print(f"Construindo grafo com limiar estrito (Threshold >= {threshold})...")
        for i in range(total_subs):
            for j in range(i + 1, total_subs):
                sub1 = subreddits[i]
                sub2 = subreddits[j]
                similarity = sim_df.iloc[i, j]
                
                if similarity >= threshold:
                    G.add_edge(sub1, sub2, weight=float(similarity))
                    
    # 4. Estratégia B: K-Nearest Neighbors (Garante que nós menores não fiquem isolados)
    else:
        print(f"Construindo grafo utilizando K-NN (k={k_neighbors})...")
        for sub in subreddits:
            # Ordenar os subreddits mais similares ignorando a si mesmo
            top_k = sim_df[sub].drop(index=sub).nlargest(k_neighbors)
            for neighbor, similarity in top_k.items():
                # No grafo não-direcionado, se o link já existe, apenas mantém
                if similarity >= threshold: # Opcional: combinar k-NN com um corte mínimo
                    G.add_edge(sub, neighbor, weight=float(similarity))
                    
    print(f"Grafo gerado: {G.number_of_nodes()} nós e {G.number_of_edges()} arestas.")
    return G

# --- Exemplo de Execução ---
df_centroids = load_centroid_data()
G = build_semantic_graph(df_centroids, threshold=0.70, k_neighbors=None)