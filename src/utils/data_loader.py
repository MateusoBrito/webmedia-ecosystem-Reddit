from pathlib import Path
import pandas as pd
import json
import os
from src.utils.paths import ROOT

DATA_RAW = ROOT / "data" / "raw" / "expansao"
DATA_PROCESSED = ROOT / "data" / "processed"
ARTIFACTS_EMBEDDINGS = ROOT / "artifacts" / "embeddings"

def load_preprocessed_data(
    only_valid_ids=True,
    columns=None
):
    """
    Carrega preprocess_text e opcionalmente filtra
    pelos ids válidos.
    """
    df = pd.read_parquet(
        DATA_PROCESSED / "preprocess_text.parquet",
        columns=columns
    )
    if only_valid_ids:
        ids_validos = pd.read_parquet(
            DATA_PROCESSED / "ids_validos.parquet"
        )
        ids = set(ids_validos["id"])
        df = df[df["id"].isin(ids)]

    return df

def load_centroid_data(
    columns=None
):
    """
    Carrega os centroides dos subreddits a partir do arquivo Parquet.
    """
    file_path = ARTIFACTS_EMBEDDINGS / "subreddit_centroids.parquet"
    
    if not file_path.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {file_path}\n"
            "Execute o script 'calculate_subreddit_centroids' primeiro."
        )

    df = pd.read_parquet(
        file_path,
        columns=columns
    )

    return df

def load_raw_data(
    columns=None,
    only_valid_ids=True
):
    """
    Carrega os dados da pasta raw/expansao
    filtra os ids validos
    """
    rows = []

    for depth in range(2):
        dir_path = DATA_RAW / str(depth) / "subreddits"
        if not dir_path.exists():
            continue
        for file_path in dir_path.glob("*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    posts = json.load(f)
                for post in posts:
                    post["depth"] = depth
                rows.extend(posts)
            except json.JSONDecodeError as e:
                print(f"Erro JSON em {file_path}: {e}")

    df = pd.DataFrame(rows)

    if only_valid_ids:
        ids_validos = pd.read_parquet(
            DATA_PROCESSED / "ids_validos.parquet"
        )
        ids = set(ids_validos["id"])
        df = df[df["id"].isin(ids)]

    if columns is not None:
        df = df[columns]
    
    print(f"Posts carregados: {len(df)}")

    return df