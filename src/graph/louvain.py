import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

from src.utils.data_loader import ROOT

# Caminhos de entrada e saída
GRAPH_PATH = ROOT / "artifacts/graph/network_disparity.graphml"
OUTPUT_PLOT = ROOT / "reports/graph_analysis/louvain_communities_macro.png"

def plot_louvain_communities():
    print('='*50)
    print(f"Carregando o grafo de:\n{GRAPH_PATH}")
    
    try:
        G = nx.read_graphml(GRAPH_PATH)
    except FileNotFoundError:
        print("Erro: Arquivo .graphml não encontrado. Rode o pipeline de extração primeiro.")
        return

    print(f"Grafo carregado com sucesso: {G.number_of_nodes()} nós e {G.number_of_edges()} arestas.")

    # --- NOVO PASSO: Extrair o Componente Gigante ---
    print("\nFiltrando ilhas isoladas (mantendo apenas o Componente Gigante)...")
    componentes = list(nx.connected_components(G))
    maior_componente = max(componentes, key=len)
    G = G.subgraph(maior_componente).copy()
    print(f"O Componente Gigante possui {G.number_of_nodes()} nós e {G.number_of_edges()} arestas.")
    print(f"Foram ignoradas {len(componentes) - 1} pequenas ilhas isoladas.")

    # 1. Aplicar o Algoritmo de Louvain com Resolução Ajustada
    print("\nExecutando clusterização de comunidades (Louvain)...")
    
    # AJUSTE AQUI: Resolução < 1.0 cria comunidades MAIORES.
    # Tente 0.8 para uma fusão leve, ou 0.5 para macro-comunidades bem grandes.
    RESOLUCAO = 0.5
    
    communities = nx.community.louvain_communities(G, weight='peso_jaccard', resolution=RESOLUCAO)
    
    # Filtrar comunidades microscópicas que sobraram (ex: com menos de 5 subreddits)
    #communities = [c for c in communities if len(c) >= 5]
    print(f"Com resolução {RESOLUCAO}, a rede foi particionada em {len(communities)} macro-comunidades válidas.")


    # 2. Mapeamento de Cores para as Comunidades
    # Precisamos criar um dicionário informando qual cor pertence a qual nó
    community_map = {}
    for i, comm in enumerate(communities):
        for node in comm:
            community_map[node] = i

    # Alinhando as cores com a ordem exata dos nós no objeto G do NetworkX
    node_colors = [community_map[node] for node in G.nodes()]

    # 3. Mapeamento de Tamanhos
    # Subreddits com muitas conexões (hubs) serão desenhados maiores
    degrees = dict(G.degree())
    node_sizes = [degrees[node] * 5 for node in G.nodes()] # Multiplicador para destacar visualmente

    # 4. Cálculo do Layout (A parte pesada)
    print("\nCalculando a disposição espacial dos nós (Spring Layout)...")
    print("Aguarde, isso pode levar de 1 a 3 minutos para redes grandes...")
    # k: Distância ótima entre os nós. iterations: Quantas vezes simular a "física" das molas
    pos = nx.spring_layout(G, k=0.15, iterations=50, seed=42)

    # 5. Plotagem Gráfica
    print("\nDesenhando e renderizando o gráfico de alta resolução...")
    # Alterado para fundo branco
    plt.figure(figsize=(18, 18), facecolor='white') 
    
    # Arestas mais escuras e ligeiramente mais visíveis para contrastar com o branco
    nx.draw_networkx_edges(G, pos, alpha=0.08, edge_color='#555555')
    
    # Desenhar os nós (Mantemos o colormap turbo, que contrasta super bem no branco)
    nx.draw_networkx_nodes(
        G, pos, 
        node_size=node_sizes, 
        cmap=plt.cm.turbo,    
        node_color=node_colors, 
        alpha=0.9,
        linewidths=0          
    )

    # Título alterado para preto
    plt.title("Estrutura Topológica do Ecossistema (Comunidades Louvain)", color='black', fontsize=24, pad=20)
    plt.axis('off') 

    # 6. Salvar e finalizar
    OUTPUT_PLOT.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    # Fundo do arquivo salvo alterado para branco
    plt.savefig(OUTPUT_PLOT, dpi=300, facecolor='white', bbox_inches='tight')
    
    print('='*50)
    print(f"Gráfico gerado com sucesso em:\n{OUTPUT_PLOT}")
    print('='*50)

if __name__ == "__main__":
    plot_louvain_communities()