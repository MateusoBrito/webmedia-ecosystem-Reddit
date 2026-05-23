from pathlib import Path
import pandas as pd
from src.utils.paths import ROOT

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