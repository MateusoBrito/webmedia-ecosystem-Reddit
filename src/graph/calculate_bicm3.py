from pathlib import Path
import networkx as nx
import pandas as pd
from bicm import BipartiteGraph
from src.utils.data_loader import load_preprocessed_data, ROOT
import os

ALPHA = 0.10
OUTPUT_PATH = Path(ROOT / "data/processed/graph_bicm.graphml")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

def load_and_categorize(columns: list[str]) -> pd.DataFrame:
    """Carrega os dados e fixa a ordem das categorias."""
    df = load_preprocessed_data(columns)

    subreddits = pd.read_parquet(Path("src/graph/teste.parquet"))
    seed = subreddits['subreddit'].tolist()
    
    # 1. Isola os autores válidos da semente
    authors = df[
        (df['subreddit'].isin(seed)) & 
        (df['author'] != '[deleted]')
    ]['author'].tolist()
    
    autores_unicos = list(set(authors))
    print(f"Quantidade de autores na semente: {len(autores_unicos):,}")
    
    # 2. Pega todas as atividades EXCLUSIVAMENTE desses autores
    atividades_dos_autores = df[df['author'].isin(autores_unicos)].copy()

    # 3. Filtro de Densidade: Mantém apenas os "super-autores" que conectam a rede
    # Conta em quantos subreddits diferentes cada um postou
    author_counts = atividades_dos_autores.groupby('author')['subreddit'].nunique()
    
    # Mantém quem postou em pelo menos 3 subreddits (você pode testar >= 2 se quiser)
    autores_conectores = author_counts[author_counts >= 3].index
    
    df_destilado = atividades_dos_autores[atividades_dos_autores['author'].isin(autores_conectores)]

    print(f"Subreddits na rede expandida: {df_destilado['subreddit'].nunique():,}")
    print(f"Autores conectores mantidos: {df_destilado['author'].nunique():,}")

    return df_destilado

def build_network_structures(df) -> tuple:
    # Cria um dicionário onde a chave é o subreddit e os valores são os usuários que postaram algo nela
    user_counts = df.groupby('subreddit')['author'].apply(set).to_dict()

    sub_list = list(user_counts.keys()) # [subreddit1, subreddit2, ..., subredditN]
    sub_idx = {s: i for i, s in enumerate(sub_list)} # {'subreddit1': 0,'subreddit2': 1}

    user_list = list({u for users in user_counts.values() for u in users}) # Conjunto com todos usuários
    user_idx = {u: i for i,u in enumerate(user_list)} # {'usuario1': 0,'usuario2': 1}


    # Cria uma lista de pares (Usuário, Subreddit) onde há uma ligação se o usuário publicou algo no subreddit
    edgelist = [
        (user_idx[u], sub_idx[s])
        for s,users in user_counts.items()
        for u in users
    ] 
    print(f"Edgelist: {len(edgelist):,} pares")

    return user_counts, sub_list, edgelist

def run_bicm(edgelist, alpha:float = ALPHA) -> list:
    # Cria a instância do BipartiteGraph e vincula sua lista de autores - subreddits
    bg = BipartiteGraph()
    bg.set_edgelist(edgelist)

    # Ele calcula: "Qual é a probabilidade do Mateus e da Maria comentarem nos mesmos dois subreddits PURAMENTE por acaso?"
    bg.solve_bicm()

    max_threads = os.cpu_count() or 4
    # Aqui nós transformamos a rede de "Usuários <-> Subreddits" em uma rede apenas de "Subreddits <-> Subreddits" (Isso é a Projeção).
    bg.compute_projection(
        rows=False, 
        alpha=alpha, 
        approx_method="normal",
        validation_method='fdr', 
        threads_num=max_threads,
        progress_bar=True)

    # O Retorno: Ele te devolve uma lista limpa: [(Subreddit 0, Subreddit 5), (Subreddit 2, Subreddit 8)].
    validated = bg.get_cols_projection(fmt='edgelist')
    print(f"\nArestas validadas (BiCM + FDR α={alpha}): {len(validated)}")
    
    return validated

def generate_graph(validated: list, users_count: dict, sub_list: list, save: bool= False):
    G = nx.Graph()
    for i,j in validated:
        s_a, s_b = sub_list[i], sub_list[j]
        u_a, u_b = users_count[s_a], users_count[s_b]
        G.add_edge(s_a,s_b, weight = len(u_a & u_b)/ len(u_a | u_b))

    isolados = [n for n in list(G.nodes()) if G.degree(n) == 0]
    G.remove_nodes_from(isolados)
    print(f"Grafo: {G.number_of_nodes()} nós, {G.number_of_edges()} arestas")

    if save:
        nx.write_graphml(G, OUTPUT_PATH)
        print(f"\nGrafo salvo com sucesso em: {OUTPUT_PATH}")

    return G

def diagnosticar_rede(df, user_counts, edgelist, sub_list):
    """Imprime estatísticas que revelam por que o BiCM retorna zero arestas."""
    
    print("=" * 60)
    print("DIAGNÓSTICO DA REDE BIPARTIDA")
    print("=" * 60)
    
    # 1. Tamanho da rede
    n_users = df['author'].nunique()
    n_subs = df['subreddit'].nunique()
    print(f"\nNós: {n_users} usuários | {n_subs} subreddits")
    print(f"Arestas: {len(edgelist):,}")
    print(f"Densidade: {len(edgelist) / (n_users * n_subs):.6f}")
    
    # 2. Distribuição de grau dos subreddits (quantos usuários por sub)
    sub_degrees = pd.Series({s: len(u) for s, u in user_counts.items()})
    print(f"\nUsuários por subreddit:")
    print(sub_degrees.describe().to_string())
    
    # 3. Distribuição de grau dos usuários (quantos subs por usuário)
    user_sub_counts = df.groupby('author')['subreddit'].nunique()
    print(f"\nSubreddits por usuário:")
    print(user_sub_counts.describe().to_string())
    
    # 4. *** A ESTATÍSTICA MAIS IMPORTANTE ***
    # Quantos pares de subreddits têm pelo menos 1 usuário em comum?
    print(f"\nCalculando overlaps entre pares de subreddits...")
    sub_names = list(user_counts.keys())
    overlaps = []
    
    # Amostra os primeiros 200 subreddits para não demorar demais
    sample = sub_names[:min(200, len(sub_names))]
    pairs_com_overlap = 0
    pairs_total = 0
    
    for i in range(len(sample)):
        for j in range(i + 1, len(sample)):
            overlap = len(user_counts[sample[i]] & user_counts[sample[j]])
            pairs_total += 1
            if overlap > 0:
                pairs_com_overlap += 1
                overlaps.append(overlap)
    
    print(f"Pares com overlap > 0: {pairs_com_overlap:,} / {pairs_total:,} ({100*pairs_com_overlap/pairs_total:.1f}%)")
    
    if overlaps:
        s = pd.Series(overlaps)
        print(f"Distribuição dos overlaps (onde > 0):")
        print(s.describe().to_string())
        print(f"\nOverlap >= 2: {(s >= 2).sum():,} pares")
        print(f"Overlap >= 5: {(s >= 5).sum():,} pares")
        print(f"Overlap >= 10: {(s >= 10).sum():,} pares")
    else:
        print(">>> PROBLEMA: NENHUM par de subreddits tem usuário em comum!")
        print(">>> O filtro está eliminando usuários de 'conexão' entre comunidades.")
    
    print("=" * 60)

if __name__ == "__main__":
    #1. Carregar os dados de cada post válido
    df = load_and_categorize(["author", "subreddit"])

    #2. Criar uma lista de (Usuário, Subreddit) para cada usuário que publicou em um subreddit
    user_counts, sub_list, edgelist = build_network_structures(df)
    diagnosticar_rede(df, user_counts, edgelist, sub_list)
    #3. Decide onde vai ter aresta de co-autoria entre dois subreddits por meio de estatística
    validated = run_bicm(edgelist)

    #4. Gera o gráfico vinculando aos nós as arestas da etapa anterior com o peso proporcional
    g = generate_graph(validated,user_counts, sub_list, save=True)
