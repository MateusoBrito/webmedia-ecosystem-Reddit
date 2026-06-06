from pathlib import Path
from typing import Tuple, List
import pandas as pd
import numpy as np

from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from umap import UMAP
from sklearn.cluster import KMeans

from src.utils.data_loader import load_preprocessed_data, load_centroid_data
from src.utils.paths import ROOT

def prepare_data():
    """
    Carrega os centroides e os textos, alinhando a ordem matemática
    dos documentos com a matriz de centroides.
    """
    print("1. Carregando e alinhando dados do centróide e posts...")
    
    # 1. Carregar textos
    df_text = load_preprocessed_data()
    df_text = df_text[df_text["depth"] == 0]
    valid_subreddits = set(df_text["subreddit"].unique())

    # 2. Carregar centroides
    df_centroids = load_centroid_data()
    df_centroids = df_centroids.loc[df_centroids.index.isin(valid_subreddits)]
    centroids_matrix = df_centroids.values
    subreddit_names = df_centroids.index.tolist()

    # 3. Agrupar os textos no nível da comunidade
    docs_per_subreddit = df_text.groupby('subreddit')['text_clean'].agg(lambda x: ' '.join(x)).reset_index()

    print(f"   -> Total de comunidades carregadas: {len(docs_per_subreddit)}")

    # 4. Reordenar os textos para garantir o match 1:1 com a matriz de centroides
    docs_per_subreddit = docs_per_subreddit.set_index('subreddit').reindex(subreddit_names).fillna("")
    documents = docs_per_subreddit['text_clean'].tolist()
    
    print(f"   -> Total de comunidades prontas: {len(documents)}")
    return documents, centroids_matrix, subreddit_names


def train_topic_model(documents: List[str], centroids_matrix: np.ndarray, n_clusters: int = 10):
    print("2. Treinando BERTopic (UMAP + KMeans + c-TF-IDF)...")
    
    umap_model = UMAP(
        n_neighbors=50,
        min_dist=0.0,
        n_components=3, 
        metric='cosine', 
        random_state=42
    )

    kmeans_model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init="auto"
    )

    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=kmeans_model,  
        ctfidf_model=ClassTfidfTransformer(),
        verbose=True
    )

    topics, _ = topic_model.fit_transform(documents, centroids_matrix)
    
    return topic_model, topics


def export_topic_dictionary(
    topic_model: BERTopic, 
    topics: List[int], 
    subreddit_names: List[str], 
    output_dir: Path
):
    """
    Extrai o dicionário semântico e mapeia quais comunidades pertencem a cada tópico.
    """
    print("3. Exportando Dicionário de Tópicos...")
    output_dir.mkdir(parents=True, exist_ok=True)

    df_topic_info = topic_model.get_topic_info()
    df_resumo = df_topic_info[['Topic', 'Count', 'Name', 'Representation']].copy()

    # Mapeamento: Quem está onde?
    df_subreddit_mapping = pd.DataFrame({
        'subreddit': subreddit_names,
        'Topic': topics
    })

    topic_to_subs = df_subreddit_mapping.groupby('Topic')['subreddit'].apply(list).reset_index()
    topic_to_subs.columns = ['Topic', 'subreddits_list']

    df_dicionario_final = df_resumo.merge(topic_to_subs, on='Topic', how='left')
    
    out_path = output_dir / "dicionario_topicos.json"
    df_dicionario_final.to_json(out_path, orient="records", indent=4)
    print(f"   -> Dicionário salvo em: {out_path}")


def export_visualizations(topic_model: BERTopic, output_dir: Path):
    """
    Gera e salva os relatórios interativos em HTML.
    """
    print("4. Gerando Visualizações HTML...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df_topic_info = topic_model.get_topic_info()
    n_clusters_total = len(df_topic_info) - 1 # Ignora o tópico -1 (outliers)

    # Gráfico de Barras (Assinaturas Linguísticas)
    fig_words = topic_model.visualize_barchart(
        top_n_topics=n_clusters_total, 
        n_words=10,
        title="Assinaturas Linguísticas do Ecossistema (c-TF-IDF)"
    )
    fig_words.write_html(output_dir / "visualizacao_topicos.html")

    # Mapa Topológico 2D
    fig_2d = topic_model.visualize_topics(
        title="Topologia do Ecossistema: Distância entre Clusters Ideológicos"
    )
    fig_2d.write_html(output_dir / "mapa_2d_topicos.html")

    print("   -> Gerando Árvore Hierárquica...")
    fig_hierarchy = topic_model.visualize_hierarchy(
        custom_labels=True, 
        title="Árvore Hierárquica dos Subreddits"
    )
    fig_hierarchy.write_html(output_dir / "hierarquia_topicos.html")
    
    print(f"   -> Visualizações salvas em: {output_dir}")


def main():
    DIR_REPORTS = ROOT / "reports" / "topic_modeling"
    
    documents, centroids_matrix, subreddit_names = prepare_data()
    topic_model, topics = train_topic_model(documents, centroids_matrix)
    
    export_topic_dictionary(topic_model, topics, subreddit_names, DIR_REPORTS)
    export_visualizations(topic_model, DIR_REPORTS)

if __name__ == "__main__":
    main()