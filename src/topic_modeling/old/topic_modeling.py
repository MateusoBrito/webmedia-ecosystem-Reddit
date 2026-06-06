from pathlib import Path
from typing import List
from itertools import product
import pandas as pd
import numpy as np
import torch
import json

from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from umap import UMAP
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics import silhouette_score
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary

from src.utils.data_loader import load_preprocessed_data
from src.utils.paths import ROOT

DIR_REPORTS = ROOT / "reports" / "topic_modeling2"
CACHE_EMBEDDINGS = ROOT / "artifacts" / "embeddings" / "embeddings_topic_modeling_all_mini.npy"
STOPWORDS_PATH = ROOT / "data" / "processed" /"stopwords.txt"


def load_stopwords(path: Path = STOPWORDS_PATH) -> set:
    with open(path, "r") as f:
        return set(line.strip() for line in f if line.strip())

def prepare_data():
    print("1. Carregando posts de profundidade 0...")

    df_full = load_preprocessed_data()
    df_full = df_full[df_full["depth"] == 0].reset_index(drop=True)

    print(f"   -> Dataset completo: {len(df_full)} posts | {df_full['subreddit'].nunique()} subreddits")

    # Remove duplicatas para o treinamento do BERTopic
    df_unique = df_full.groupby(by="text_clean").aggregate("id")
    print(f"   -> Textos ÚNICOS para treinamento: {len(df_unique)}")

    documents_unique = df_unique["text_clean"].tolist()
    
    return documents_unique, df_unique

def generate_embeddings(
    documents: List[str],
    cache_path: Path = CACHE_EMBEDDINGS,
    model_name: str = "all-MiniLM-L6-v2"
) -> np.ndarray:
    print("2. Gerando embeddings...")

    if cache_path.exists():
        print(f"   -> Carregando da cache: {cache_path}")
        return np.load(cache_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   -> Dispositivo: {device}")

    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(
        documents,
        batch_size=128,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, embeddings)
    print(f"   -> Cache salva em: {cache_path}")

    return embeddings


def train_topic_model(
    documents: List[str],
    embeddings: np.ndarray,
    stopwords: set,
    n_clusters: int = 10,
    n_neighbors: int = 50,
    n_components: int = 3
):
    print("1. Treinando BERTopic (UMAP + KMeans + c-TF-IDF)...")

    umap_model = UMAP(
        n_neighbors= n_neighbors,
        min_dist=0.0,
        n_components= n_components,
        metric='cosine',
        random_state=42
    )

    kmeans_model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init="auto"
    )

    vectorizer = CountVectorizer(stop_words=list(stopwords))
    
    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=kmeans_model,
        ctfidf_model=ClassTfidfTransformer(),
        vectorizer_model=vectorizer,
        verbose=True
    )

    topics, _ = topic_model.fit_transform(documents, embeddings)

    return topic_model, topics

def evaluate_model(
    topic_model: BERTopic,
    topics: List[int],
    documents: List[str],
    embeddings: np.ndarray
) -> dict:
    print("2. Avaliando modelo...")

    # --- Silhoueta ---
    silhouette = silhouette_score(embeddings, topics, metric="cosine")
    print(f"   -> Silhoueta: {silhouette:.4f}")

    # --- Diversidade ---
    topic_words = []
    for t in set(topics):
        if t == -1:
            continue
        words = topic_model.get_topic(t)
        if not words:
            continue
        topic_words.append([word for word, _ in words])
    all_words = [word for words in topic_words for word in words]
    diversity = len(set(all_words)) / len(all_words) if all_words else 0.0
    print(f"   -> Diversidade: {diversity:.4f}")

    # --- Coerência (c_v) ---
    tokenized = [doc.split() for doc in documents]
    dictionary = Dictionary(tokenized)
    coherence_model = CoherenceModel(
        topics=topic_words,
        texts=tokenized,
        dictionary=dictionary,
        coherence="c_v"
    )
    coherence = coherence_model.get_coherence()
    print(f"   -> Coerência (c_v): {coherence:.4f}")

    return {
        "n_topics": len(set(topics)) - (1 if -1 in topics else 0),
        "silhouette": round(silhouette, 4),
        "diversity": round(diversity, 4),
        "coherence_cv": round(coherence, 4),
    }

def grid_search(
    documents: List[str],
    embeddings: np.ndarray,
    stopwords: list,
    output_dir: Path
) -> pd.DataFrame:
    print("Grid search BERTopic...")

    param_grid = {
        "n_clusters":  [5, 10, 15, 20, 30],
        "n_neighbors": [15, 30, 50],
        "n_components": [3, 5],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "grid_search_results.csv"

    keys = list(param_grid.keys())
    combinations = list(product(*param_grid.values()))
    print(f"   -> {len(combinations)} combinações")

    if out_path.exists():
        df_done = pd.read_csv(out_path)
        done_params = set(
            tuple(row[k] for k in keys)
            for _, row in df_done.iterrows()
        )
        print(f"   -> {len(done_params)} combinações já concluídas, retomando...")
    else:
        df_done = pd.DataFrame()
        done_params = set()

    for i, values in enumerate(combinations):
        params = dict(zip(keys, values))

        if tuple(values) in done_params:
            print(f"[{i+1}/{len(combinations)}] Pulando {params}")
            continue

        print(f"\n[{i+1}/{len(combinations)}] {params}")

        try:
            topic_model, topics = train_topic_model(
                documents,
                embeddings,
                stopwords,
                n_clusters=params["n_clusters"],
                n_neighbors=params["n_neighbors"],
                n_components=params["n_components"],
            )

            metrics = evaluate_model(topic_model, topics, documents, embeddings)

            n_outliers = sum(1 for t in topics if t == -1)
            outlier_pct = round(n_outliers / len(topics) * 100, 2)

            row = {**params, **metrics, "outlier_pct": outlier_pct}

        except Exception as e:
            print(f"   -> Falhou: {e}")
            row = {**params, "error": str(e)}

        df_done = pd.concat([df_done, pd.DataFrame([row])], ignore_index=True)
        df_done.to_csv(out_path, index=False)

    return df_done

def export_topic_dictionary(
    topic_model: BERTopic,
    topics: List[int],
    df: pd.DataFrame,
    output_dir: Path
):
    print("3. Exportando Dicionário de Tópicos...")
    output_dir.mkdir(parents=True, exist_ok=True)

    df_topic_info = topic_model.get_topic_info()
    df_resumo = df_topic_info[['Topic', 'Count', 'Name', 'Representation']].copy()
    df_resumo.to_json(output_dir / "topic_info.json", orient="records", indent=4)

    df_mapping = df[["id"]].copy()
    df_mapping["topic"] = topics
    df_mapping.to_parquet(output_dir / "post_topics.parquet", index=False)

    print(f"   -> Salvo em: {output_dir}")


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

def select_best_configs(df: pd.DataFrame) -> dict:
    df = df.dropna(subset=["silhouette", "diversity", "coherence_cv"])

    # melhor de cada métrica individual
    best_silhouette = df.loc[df["silhouette"].idxmax()]
    best_diversity  = df.loc[df["diversity"].idxmax()]
    best_coherence  = df.loc[df["coherence_cv"].idxmax()]

    # melhor média normalizada
    for col in ["silhouette", "diversity", "coherence_cv"]:
        min_v, max_v = df[col].min(), df[col].max()
        df[f"{col}_norm"] = (df[col] - min_v) / (max_v - min_v)

    df["score"] = df[["silhouette_norm", "diversity_norm", "coherence_cv_norm"]].mean(axis=1)
    best_combined = df.loc[df["score"].idxmax()]

    return {
        "best_silhouette": best_silhouette,
        "best_diversity":  best_diversity,
        "best_coherence":  best_coherence,
        "best_combined":   best_combined,
    }


def run_best_models(
    documents: list,
    embeddings: np.ndarray,
    df_grid: pd.DataFrame,
    df: pd.DataFrame,
    stopwords: list,
    output_dir: Path
):
    param_keys = ["n_clusters", "n_neighbors", "n_components"]
    best_configs = select_best_configs(df_grid)

    for name, row in best_configs.items():
        print(f"\n{'='*50}")
        print(f"Rodando: {name}")

        params = {k: int(row[k]) for k in param_keys}
        print(f"   -> Params: {params}")

        model_dir = output_dir / name
        model_dir.mkdir(parents=True, exist_ok=True)

        topic_model, topics = train_topic_model(documents, embeddings, stopwords ,**params)
        metrics = evaluate_model(topic_model, topics, documents, embeddings)

        n_outliers = sum(1 for t in topics if t == -1)
        metrics["outlier_pct"] = round(n_outliers / len(topics) * 100, 2)
        metrics["params"] = params

        with open(model_dir / "metricas.json", "w") as f:
            json.dump(metrics, f, indent=4)

        export_topic_dictionary(topic_model, topics, df, model_dir)
        export_visualizations(topic_model, model_dir)

        print(f"   -> Salvo em: {model_dir}")

def main():
    stopwords = load_stopwords(STOPWORDS_PATH)
    documents, df = prepare_data()
    embeddings = generate_embeddings(documents, cache_path=CACHE_EMBEDDINGS)
    df_results = grid_search(documents, embeddings, stopwords, DIR_REPORTS)

    run_best_models(documents, embeddings, df_results, df, stopwords ,DIR_REPORTS)

if __name__ == "__main__":
    main()