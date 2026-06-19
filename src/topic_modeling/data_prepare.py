from src.utils.data_loader import load_preprocessed_data, ROOT
from pathlib import Path
import pandas as pd

def load_stopwords(path: Path) -> set:
    with open(path, "r") as f:
        return set(line.strip() for line in f if line.strip())

def prepare_data(filtro_alvo: str = "depth_0"):
    print("1. Carregando posts...")
    
    if filtro_alvo == "depth_0":
        df_full = load_preprocessed_data(columns=["id", "text","text_clean","depth","subreddit"])
        df_full = df_full[df_full["depth"] == 0].reset_index(drop=True)
    
    elif filtro_alvo == "rede_expandida":
        # 1. Carrega a lista dos subreddits de destino
        caminho_subs = Path(ROOT / "data" / "processed" / "subreddits_contra_expanded_network.parquet")
        if not caminho_subs.exists():
            raise FileNotFoundError(f"Arquivo de subreddits não encontrado em: {caminho_subs}")
            
        df_subs = pd.read_parquet(caminho_subs)
        subreddits_expandidos = df_subs["subreddit"].tolist()
        
        # 2. Carrega a base completa e filtra para manter APENAS os posts desses subreddits novos
        df_full = load_preprocessed_data(columns=["id", "text", "text_clean", "depth", "subreddit"])
        df_full = df_full[df_full["subreddit"].isin(subreddits_expandidos)].reset_index(drop=True)
        
        print(f"   -> Dataset da Rede Expandida (Zonas de Contágio) carregado.")
        print(f"   -> Foram encontrados {len(df_full)} posts provenientes de {len(subreddits_expandidos)} subreddits periféricos.")
    else:
        raise ValueError("Filtro alvo inválido. Use 'depth_0' ou 'rede_expandida'.")

    print(f"   -> Dataset completo: {len(df_full)} posts")

    df_unique = df_full.drop_duplicates(subset=["text_clean"]).reset_index(drop=True)
    documents_unique = df_unique["text_clean"].tolist()
    print(f"   -> Textos ÚNICOS para treinamento: {len(df_unique)}")
    
    return documents_unique, df_unique, df_full