from typing import List
from pathlib import Path
from itertools import product
import pandas as pd
import numpy as np
#import joblib

from src.topic_modeling.model import train_topic_model, evaluate_model

def grid_search(
    documents: List[str],
    embeddings: np.ndarray,
    stopwords: set, 
    param_grid: dict, 
    output_dir: Path
) -> pd.DataFrame:
    print("Grid search BERTopic...")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "grid_search_results.csv"

    #temp_dir = output_dir / "temp_models_cache"
    #temp_dir.mkdir(parents=True, exist_ok=True)

    keys = list(param_grid.keys())
    combinations = list(product(*param_grid.values()))
    print(f"   -> {len(combinations)} combinações")

    done_params = set()
    
    if out_path.exists():
        df_done = pd.read_csv(out_path)
        done_params = set(
            tuple(row[k] for k in keys)
            for _, row in df_done.iterrows()
        )
        print(f"   -> {len(done_params)} combinações já concluídas, retomando...")

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

            #model_id = f"c{params['n_clusters']}_n{params['n_neighbors']}_comp{params['n_components']}"
            #arquivo_cache = temp_dir / f"{model_id}.joblib"
            #joblib.dump((topic_model, topics), arquivo_cache)

            metrics = evaluate_model(topic_model, topics, documents, embeddings)

            n_outliers = sum(1 for t in topics if t == -1)
            outlier_pct = round(n_outliers / len(topics) * 100, 2)

            row = {**params, **metrics, "outlier_pct": outlier_pct}

        except Exception as e:
            print(f"   -> Falhou: {e}")
            row = {**params, "error": str(e)}

        df_row = pd.DataFrame([row])
        df_row.to_csv(out_path, mode='a', header=not out_path.exists(), index=False)

    return pd.read_csv(out_path)

def select_best_configs(df: pd.DataFrame, top_n: int = 1) -> dict:
    # 1. Remove linhas que falharam
    df = df.dropna(subset=["silhouette", "diversity", "coherence_cv"]).copy()

    # 2. Cria métricas normalizadas (0 a 1) para encontrar a melhor média
    for col in ["silhouette", "diversity", "coherence_cv"]:
        min_v, max_v = df[col].min(), df[col].max()
        if max_v > min_v:
            df[f"{col}_norm"] = (df[col] - min_v) / (max_v - min_v)
        else:
            df[f"{col}_norm"] = 0.0

    # 3. Calcula os scores combinados
    df["score_sdc"] = df[["silhouette_norm", "diversity_norm", "coherence_cv_norm"]].mean(axis=1)
    df["score_sc"] = df[["silhouette_norm", "coherence_cv_norm"]].mean(axis=1)

    # 4. Mapeia o nome base do dicionário para a coluna correspondente no DataFrame
    metricas_alvo = {
        "silhouette": "silhouette",
        "diversity": "diversity",
        "coherence": "coherence_cv",
        "combined_sdc": "score_sdc",
        "combined_sc": "score_sc"
    }

    best_configs = {}

    # 5. Extrai os Top N de cada métrica e adiciona ao dicionário
    for nome_metrica, coluna_df in metricas_alvo.items():
        # Pega as N melhores linhas baseadas na coluna atual
        top_df = df.nlargest(top_n, coluna_df)
        
        # Itera sobre os vencedores para salvar no dicionário com o ranking no nome
        for rank, (_, row) in enumerate(top_df.iterrows(), start=1):
            chave = f"best_{nome_metrica}_rank{rank}"
            best_configs[chave] = row

    return best_configs