from pathlib import Path
import numpy as np
import torch
import argparse
import networkx as nx
from sentence_transformers import SentenceTransformer
from src.utils.data_loader import load_preprocessed_data

def run_embeddings(
    root=None,
    output_path=None,
    model_name="paraphrase-multilingual-MiniLM-L12-v2",
    batch_size=128
):
    if root is None:
        root = Path(__file__).resolve().parents[3]
    
    dir_path = root / "artifacts" / "embeddings"
    if output_path is None:
        output_path = dir_path / "embeddings_posts.npz"

    dir_path.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        print("Embeddings já existem. Pulando execução.")
        return
    
    graph_path = root / "artifacts" / "graph" / "network_disparity.graphml"
    print(f"Carregando grafo de {graph_path}...")
    try:
        G = nx.read_graphml(graph_path)
        valid_subreddits = set(G.nodes())
        print(f"-> {len(valid_subreddits)} subreddits válidos encontrados na topologia.")
    except FileNotFoundError:
        print(f"Erro: O grafo não foi encontrado no caminho {graph_path}.")
        print("Execute a filtragem estatística antes de gerar os embeddings.")
        return

    df = load_preprocessed_data()
    df = df[df['subreddit'].isin(valid_subreddits)].copy()
    print(f"-> Total de posts APÓS o filtro do grafo: {len(df)}")

    sentences = df["text_clean"].tolist()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device}")

    model = SentenceTransformer(model_name, device=device)

    print(f"Processando {len(sentences)} textos...")

    embeddings = model.encode(
        sentences,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    print(f"Salvando em {output_path}")
    np.savez(
        output_path,
        embeddings=embeddings,
        ids=df["id"].values
    )

    print("Concluído!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--model", type=str, default="paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--batch_size", type=int, default=128)

    args = parser.parse_args()

    run_embeddings(
        output_path=Path(args.output),
        model_name=args.model,
        batch_size=args.batch_size
    )