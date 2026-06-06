import json
import random
import argparse
import pandas as pd
from pathlib import Path
from topic_modeling.old.topic_modeling import (
    prepare_data, generate_embeddings, train_topic_model, load_stopwords
)
from src.utils.paths import ROOT


def load_model_artifacts(model_dir: Path) -> dict:
    with open(model_dir / "metricas.json") as f:
        return json.load(f)


def sample_topic_docs(
    topic_model,
    topics: list,
    df: pd.DataFrame,
    n_representative: int = 3,
    n_random: int = 7,
    seed: int = 42
) -> list:
    random.seed(seed)
    result = []

    df_mapping = df[["id", "text_clean"]].copy()
    df_mapping["topic"] = topics

    for t in sorted(set(topics)):
        if t == -1:
            continue

        words = [word for word, _ in (topic_model.get_topic(t) or [])]
        rep_texts = topic_model.get_representative_docs(t)[:n_representative]

        topic_docs = df_mapping[df_mapping["topic"] == t]

        rep = topic_docs[topic_docs["text_clean"].isin(rep_texts)] # Limitar apenas 3 (tem textos repetidos)
        rep_ids = rep["id"].tolist()
        rep_texts_found = rep["text_clean"].tolist()

        pool = topic_docs[~topic_docs["id"].isin(rep_ids)]
        rand_sample = pool.sample(min(n_random, len(pool)), random_state=seed)

        result.append({
            "topic": t,
            "count": len(topic_docs),
            "words": words,
            "representative": [{"id": i, "text": t} for i, t in zip(rep_ids, rep_texts_found)],
            "random": [{"id": i, "text": t} for i, t in zip(rand_sample["id"], rand_sample["text_clean"])],
        })

    return result


def build_summarization_input(metrics: dict, samples: list, output_path: Path):
    output = {
        "metrics": metrics,
        "topics": [
            {
                "topic_id": s["topic"],
                "count": s["count"],
                "top_words": s["words"],
                "representative_posts": s["representative"],
                "random_posts": s["random"],
            }
            for s in samples
        ]
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)

    print(f"   -> Salvo em: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    output_path = Path(args.output) if args.output else model_dir / "summarization_input.json"

    print(f"Lendo métricas de: {model_dir}")
    metrics = load_model_artifacts(model_dir)
    params = metrics["params"]
    print(f"Parâmetros: {params}")

    stopwords = load_stopwords()
    documents, df = prepare_data()
    embeddings = generate_embeddings(documents)

    print("Retreinando modelo com os parâmetros salvos...")
    topic_model, topics = train_topic_model(
        documents,
        embeddings,
        stopwords,
        n_clusters=params["n_clusters"],
        n_neighbors=params["n_neighbors"],
        n_components=params["n_components"],
    )

    print("Amostrando documentos por tópico...")
    samples = sample_topic_docs(topic_model, topics, df)

    print("Construindo arquivo de sumarização...")
    build_summarization_input(metrics, samples, output_path)

    print("Concluído!")


if __name__ == "__main__":
    main()