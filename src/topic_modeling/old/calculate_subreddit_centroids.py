from pathlib import Path
import pandas as pd
import numpy as np
import argparse
from numpy.linalg import norm
from src.utils.data_loader import load_preprocessed_data

def load_data(input_embeddings):
    print("Carregando dados...")
    df = load_preprocessed_data()

    data = np.load(input_embeddings, allow_pickle=True)
    embeddings = data["embeddings"]
    ids = data["ids"] 
    
    id_to_idx = {id_: i for i, id_ in enumerate(ids)}
    df = df.set_index("id").loc[ids].reset_index()

    if len(df) != embeddings.shape[0]:
        raise ValueError(f"Mismatch: df tem {len(df)} linhas, embeddings tem {embeddings.shape[0]}")

    return df, embeddings

def compute_centroids(df, embeddings):
    df = df.reset_index(drop=True)
    groups = df.groupby("subreddit").indices

    centroids = {}

    for sub, idx in groups.items():
        vecs = embeddings[idx]
        c = vecs.mean(axis=0)
        centroids[sub] = c / norm(c) 

    return centroids


def save_centroids(centroids, output_path):
    df_centroids = pd.DataFrame.from_dict(centroids, orient="index")
    df_centroids.index.name = "subreddit"

    df_centroids.to_parquet(output_path, index=True)
    print(f"Salvo em: {output_path}")


def main(input_embeddings, output_path):
    df, embeddings = load_data(input_embeddings)
    centroids = compute_centroids(df, embeddings)
    save_centroids(centroids, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--embeddings", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)

    args = parser.parse_args()

    main(Path(args.embeddings), Path(args.output))