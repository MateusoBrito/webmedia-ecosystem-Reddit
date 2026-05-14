import os
import json
import pandas as pd
from transformers import pipeline
import torch
from tqdm import tqdm

# ===================================== LENDO DATASET =====================================
def read_dataset():
    rows = []
    erros = []

    # Itera por cada profundidade
    for depth in range(3 + 1):
        dir_ = f"../data/raw/expansao/{depth}/subreddits"
        if not os.path.exists(dir_):
            continue
        for filename in os.listdir(dir_):
            if filename.endswith(".json"):
                path = os.path.join(dir_, filename)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        posts = json.load(f)
                    for post in posts:
                        post["depth"] = depth
                    rows.extend(posts)
                except json.JSONDecodeError:
                    erros.append(path)

    df = pd.DataFrame(rows)
    print(f"Quantidade de posts carregados: {len(df)}")
    return df

# ===================================== CONCATENANDO E FILTRANDO =====================================

def filter_df(df,checkpoint_path):
    df['text'] = (df["title"].fillna("") + " " + df["selftext"].fillna("")).str.strip()

    # Filtra textos vazios
    mask_validos = df['text'].str.strip().str.len() > 0
    df = df[mask_validos].copy()

    print(f"Após filtrar textos vazios {len(df)}")

    # Não executa o que já foi processado
    if os.path.exists(checkpoint_path):
        checkpoint = pd.read_csv(checkpoint_path)
        ids_processados = set(checkpoint['id'].tolist())
        df = df[~df['id'].isin(ids_processados)]
        print(f"Quantidade já classificados: {len(ids_processados)}. Faltante: {len(df)}")
    
    return df

# ===================================== SEGUNDA CLASSIFICAÇÃO =====================================

def run_classificacao(df, checkpoint_path):
    device = 0 if torch.cuda.is_available() else -1
    print(f"Usando: {'GPU' if device == 0 else 'CPU'}")

    model_id = "papluca/xlm-roberta-base-language-detection"
    lang_classifier = pipeline("text-classification", model=model_id, device=device)

    tokenizer = lang_classifier.tokenizer
    max_tokens = tokenizer.model_max_length

    print(f"O limite nativo deste modelo é de {max_tokens} tokens.")

    texts = df['text'].astype(str).tolist()

    batch_size = 128
    print(f"Classificando {len(texts)} posts com RoBERTa...")

    for i in tqdm(range(0,len(texts), batch_size)):
        df_batch = df.iloc[i : i + batch_size].copy()
        batch_texts = df_batch['text'].astype(str).tolist()

        try:
            outputs = lang_classifier(batch_texts, truncation=True, max_length=max_tokens)
            df_batch['lang_roberta'] = [out['label'] for out in outputs]
            df_batch[['id', 'lang', 'lang_roberta']].to_csv(
                checkpoint_path, 
                mode='a', 
                index=False, 
                header=not os.path.exists(checkpoint_path)
            )
        except Exception as e:
            print(f"\nErro no batch {i}: {e}")
            break

    print(f"\nProcessamento concluído. Resultados salvos em: {checkpoint_path}")

if __name__ == "__main__":
    OUTPUT = "../reports/lang_detection.csv"
    
    df = read_dataset()
    df = filter_df(df, OUTPUT)
    run_classificacao(df, OUTPUT)
