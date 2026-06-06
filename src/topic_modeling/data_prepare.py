from src.utils.data_loader import load_preprocessed_data
from pathlib import Path

def load_stopwords(path: Path) -> set:
    with open(path, "r") as f:
        return set(line.strip() for line in f if line.strip())

def prepare_data():
    print("1. Carregando posts...")
    df_full = load_preprocessed_data(columns=["id", "text","text_clean","depth"])
    df_full = df_full[df_full["depth"] == 0].reset_index(drop=True)
    print(f"   -> Dataset completo: {len(df_full)} posts")

    # Apenas remove as duplicatas, sem precisar fazer agrupamento de listas!
    df_unique = df_full.drop_duplicates(subset=["text_clean"]).reset_index(drop=True)
    documents_unique = df_unique["text_clean"].tolist()
    print(f"   -> Textos ÚNICOS para treinamento: {len(df_unique)}")
    
    return documents_unique, df_unique, df_full