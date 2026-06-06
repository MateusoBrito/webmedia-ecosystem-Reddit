import yaml
import json
import argparse 
import pandas as pd
import numpy as np
from pathlib import Path

from src.utils.paths import ROOT 

from src.topic_modeling.data_prepare import load_stopwords, prepare_data
from src.topic_modeling.embeddings import generate_embeddings
from src.topic_modeling.optimization import grid_search, select_best_configs
from src.topic_modeling.model import train_topic_model, evaluate_model
from src.topic_modeling.export import export_topic_dictionary, export_visualizations


def load_config(config_path: str = "config.yaml") -> dict:
    """Lê o arquivo YAML e retorna um dicionário."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def run_best_models(
    documents_unique: list, 
    embeddings: np.ndarray, 
    df_grid: pd.DataFrame, 
    df_unique: pd.DataFrame, 
    df_full: pd.DataFrame, 
    stopwords: set, 
    output_dir: Path,
    top_n: int = 1
):
    best_configs = select_best_configs(df_grid, top_n=top_n)
    param_keys = ["n_clusters", "n_neighbors", "n_components"]

    # 1. Agrupa os modelos únicos, mas agora listando todas as PASTAS de destino
    modelos_para_treinar = {}
    for caminho_relativo, row in best_configs.items():
        params_tupla = tuple(int(row[k]) for k in param_keys)
        
        if params_tupla not in modelos_para_treinar:
            modelos_para_treinar[params_tupla] = {
                "params": {k: int(row[k]) for k in param_keys},
                "destinos": [caminho_relativo] # Inicia a lista de pastas
            }
        else:
            modelos_para_treinar[params_tupla]["destinos"].append(caminho_relativo)

    total_modelos = len(modelos_para_treinar)
    print(f"\nFiltro Inteligente: {total_modelos} treinamentos cobrirão {len(best_configs)} rankings.")

    # 2. Loop de Treinamento
    for i, (params_tupla, info) in enumerate(modelos_para_treinar.items(), start=1):
        params = info["params"]
        destinos = info["destinos"] # Ex: ["best_combined_sc/top1", "best_silhouette/top2"]
        
        print(f"\n{'='*60}")
        print(f"[{i}/{total_modelos}] Treinando config: Clusters={params['n_clusters']} | UMAP={params['n_neighbors']}")
        print(f"-> Este modelo será salvo nas seguintes pastas:")
        for d in destinos:
            print(f"   - {d}")
        print(f"{'='*60}")
        
        # A. Treino Único (Onde a GPU trabalha)
        topic_model, topics = train_topic_model(documents_unique, embeddings, stopwords, **params)
        
        # B. Avaliação Única
        metrics = evaluate_model(topic_model, topics, documents_unique, embeddings)
        metrics["params"] = params

        # C. Exportação Múltipla (Salva rapidamente em todas as pastas que este modelo ganhou)
        for caminho in destinos:
            # Junta "topic_modeling5" + "best_combined_sc" + "top1"
            model_dir = output_dir / caminho 
            model_dir.mkdir(parents=True, exist_ok=True)

            # Salva o JSON de métricas
            with open(model_dir / "metricas.json", "w") as f:
                json.dump(metrics, f, indent=4)

            # Exporta dicionário e gráficos
            export_topic_dictionary(topic_model, topics, df_unique, df_full, model_dir)
            export_visualizations(topic_model, model_dir)
            
            print(f"   [✓] Salvo em: {model_dir}")


def main():
    parser = argparse.ArgumentParser(description="Pipeline de Topic Modeling")
    parser.add_argument(
        "--config", 
        "-c", 
        type=str, 
        default="config.yaml",
        help="Nome do arquivo de configuração YAML (ex: config_teste.yaml)"
    )

    args = parser.parse_args()

    # 2. Carrega as configurações usando o argumento passado
    config_path = ROOT / args.config
    print(f"Carregando configurações de: {config_path}")
    config = load_config(config_path)
    
    stopwords_path = ROOT / config["paths"]["stopwords"]
    cache_embeddings = ROOT / config["paths"]["cache_embeddings"]
    dir_reports = ROOT / config["paths"]["dir_reports"]

    # 2. Prepara os dados
    stopwords = load_stopwords(stopwords_path)
    documents_unique, df_unique, df_full = prepare_data()
    
    # 3. Gera os Embeddings passando os parâmetros dinâmicos
    embeddings = generate_embeddings(
        documents_unique, 
        cache_path=cache_embeddings,
        model_name=config["model"]["embedding_name"]
    )
    
    # 4. Executa a busca de hiperparâmetros
    df_results = grid_search(
        documents_unique, 
        embeddings, 
        stopwords, 
        param_grid=config["grid_search"], 
        output_dir=dir_reports
    )

    # 5. Treina e exporta os melhores modelos encontrados
    run_best_models(
        documents_unique, 
        embeddings, 
        df_results, 
        df_unique, 
        df_full, 
        stopwords,
        top_n = config["paths"]["top_n"],
        output_dir=dir_reports
    )

if __name__ == "__main__":
    main()