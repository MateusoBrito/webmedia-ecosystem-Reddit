import argparse
import random
import json
import pandas as pd
from pathlib import Path

# Importando do seu ecossistema
from src.utils.paths import ROOT
from src.topic_modeling.data_prepare import load_stopwords, prepare_data
from src.topic_modeling.embeddings import generate_embeddings
from src.topic_modeling.model import train_topic_model

CACHE_EMBEDDINGS = Path("artifacts/embeddings/embeddings_topic_modeling_all_mini.npy")
STOPWORDS_PATH = Path("data/processed/stopwords.txt")

def format_topic_data(
    topic_id: int, 
    top_words: list, 
    rep_docs: list, 
    sample_docs: list
) -> str:
    """Formata apenas o bloco de dados brutos de um tópico específico."""
    words_str = ", ".join([word for word, _ in top_words])
    
    # Tratamento caso alguma lista venha vazia
    rep_docs_str = "\n".join([f"  - {doc}" for doc in rep_docs]) if rep_docs else "  - [Nenhum texto representativo]"
    sample_docs_str = "\n".join([f"  - {doc}" for doc in sample_docs]) if sample_docs else "  - [Nenhuma amostra adicional]"

    return f"""### TÓPICO {topic_id} ###
top_10_words: [{words_str}]

top_3_transcriptions: 
{rep_docs_str}

sample_transcriptions: 
{sample_docs_str}
--------------------------------------------------------------------------------
"""

def generate_summarization_prompts(model_dir: Path):
    print(f"\n1. Carregando configurações do modelo de: {model_dir}")
    
    with open(model_dir / "metricas.json", "r") as f:
        metrics = json.load(f)
        params = metrics["params"]
        
    print(f"   -> Parâmetros recuperados: {params}")

    # B. Prepara os dados (Carregando a coluna original 'text' também)
    stopwords = load_stopwords(STOPWORDS_PATH)
    documents_unique, df_unique, df_full = prepare_data()
    embeddings = generate_embeddings(documents_unique, cache_path=CACHE_EMBEDDINGS)
    
    # C. Retreina para pegar representativos oficiais
    print("\n3. Retreinando modelo para extrair os textos representativos matemáticos...")
    topic_model, topics = train_topic_model(
        documents=documents_unique,
        embeddings=embeddings,
        stopwords=stopwords,
        n_clusters=params["n_clusters"],
        n_neighbors=params["n_neighbors"],
        n_components=params["n_components"]
    )

    # D. Recria o mapeamento
    topic_mapping = dict(zip(df_unique["text_clean"], topics))
    df_full["topic"] = df_full["text_clean"].map(topic_mapping)

    # Arquivo de saída apenas com os dados
    out_file = model_dir / "dados_topicos_sumarizacao.txt"
    valid_topics = [t for t in set(topics) if t != -1]
    
    print(f"\n4. Gerando arquivo de dados para {len(valid_topics)} tópicos válidos...")
    
    with open(out_file, "w", encoding="utf-8") as f:
        for t in sorted(valid_topics):
            
            # 1. Palavras-chave
            top_words = topic_model.get_topic(t)[:10]
            
            # 2. Dicionário tradutor (Limpo -> Original)
            # Obs: Altere 'text' para o nome exato da sua coluna original se necessário
            df_topico = df_full[df_full['topic'] == t].dropna(subset=['text_clean'])
            tradutor_para_original = dict(zip(df_topico["text_clean"], df_topico["text"]))
            
            # 3. Textos Representativos TRADUZIDOS para o original
            rep_docs_limpos = topic_model.get_representative_docs(t)
            rep_docs_limpos = rep_docs_limpos[:3] if rep_docs_limpos else []
            rep_docs = [tradutor_para_original.get(doc, doc) for doc in rep_docs_limpos]
            
            # 4. Amostras Aleatórias (Direto da coluna original)
            pool_amostra_original = df_topico["text"].dropna().drop_duplicates().tolist()
            
            # Remove os representativos do pool para não haver duplicidade na leitura do LLM
            pool_amostra = [doc for doc in pool_amostra_original if doc not in rep_docs]
            
            n_samples = min(7, len(pool_amostra))
            sample_docs = random.sample(pool_amostra, n_samples) if n_samples > 0 else []
            
            # Formata os dados deste tópico e escreve no arquivo
            topic_data_text = format_topic_data(t, top_words, rep_docs, sample_docs)
            f.write(topic_data_text)
            f.write("\n")

    print(f"-> Sucesso! Arquivo de dados unificado gerado em: {out_file}")

def main():
    parser = argparse.ArgumentParser(description="Gera dados puros de tópicos para sumarização")
    parser.add_argument(
        "--model_dir", 
        "-m", 
        type=str, 
        required=True,
        help="Caminho relativo para a pasta do modelo (ex: reports/topic_modeling/best_combined)"
    )
    args = parser.parse_args()

    model_dir_path = ROOT / args.model_dir

    if not (model_dir_path / "metricas.json").exists():
        print(f"ERRO: Não encontrei o metricas.json na pasta: {model_dir_path}")
        return

    generate_summarization_prompts(model_dir_path)

if __name__ == "__main__":
    main()
