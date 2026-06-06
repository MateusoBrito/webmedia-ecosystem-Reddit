from typing import List
import numpy as np
from bertopic import BERTopic
from pathlib import Path
import pandas as pd
import random
import json

def export_topic_dictionary(
    topic_model: BERTopic,
    topics: List[int],
    df_unique: pd.DataFrame, 
    df_full: pd.DataFrame,
    output_dir: Path
):
    print("3. Exportando Dicionário de Tópicos Enriquecido...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Cria o mapa base ligando o texto limpo ao número do tópico
    topic_mapping = dict(zip(df_unique["text_clean"], topics))
    
    # 2. Anexa temporariamente o tópico ao df_full para conseguirmos puxar os textos originais
    df_full_temp = df_full.copy()
    df_full_temp["topic"] = df_full_temp["text_clean"].map(topic_mapping)

    # 3. Pega a tabela de informações básicas do BERTopic
    df_topic_info = topic_model.get_topic_info()
    topics_data = []

    # 4. Constrói o JSON linha a linha, anexando os textos representativos e aleatórios
    for _, row in df_topic_info.iterrows():
        t = row['Topic']
        
        # Estrutura base do tópico
        topic_dict = {
            "Topic": t,
            "Count": row["Count"],
            "Name": row["Name"],
            "Representation": row["Representation"],
            "Representative_Docs": [],
            "Sample_Docs": []
        }

        # Ignoramos a amostragem para o tópico -1 (Outliers) para não poluir o LLM
        if t != -1:
            # Filtra apenas os posts originais deste tópico específico
            # ATENÇÃO: Confirme se a coluna do texto original no seu df_full se chama 'text'
            df_topico = df_full_temp[df_full_temp['topic'] == t].dropna(subset=['text_clean'])
            
            # Cria o tradutor: Texto Limpo (BERTopic) -> Texto Original (LLM)
            tradutor = dict(zip(df_topico["text_clean"], df_topico["text"]))

            # A. Extrai os 3 Documentos Representativos oficiais e traduz
            rep_docs_limpos = topic_model.get_representative_docs(t)
            rep_docs_limpos = rep_docs_limpos[:3] if rep_docs_limpos else []
            rep_docs_originais = [tradutor.get(doc, doc) for doc in rep_docs_limpos]
            
            topic_dict["Representative_Docs"] = rep_docs_originais

            # B. Extrai 7 Amostras Aleatórias únicas
            pool_amostra_original = df_topico["text"].dropna().drop_duplicates().tolist()
            
            # Remove os representativos do pool para evitar que o LLM leia mensagens duplicadas
            pool_amostra = [doc for doc in pool_amostra_original if doc not in rep_docs_originais]
            
            n_samples = min(7, len(pool_amostra))
            sample_docs = random.sample(pool_amostra, n_samples) if n_samples > 0 else []
            
            topic_dict["Sample_Docs"] = sample_docs

        topics_data.append(topic_dict)

    # 5. Salva o JSON final super formatado
    with open(output_dir / "topic_info.json", "w", encoding="utf-8") as f:
        json.dump(topics_data, f, indent=4, ensure_ascii=False)

    # 6. Salva o Parquet para mapeamento de rede/estatística (Mantido idêntico)
    df_mapping = df_full[["id", "text_clean"]].copy()
    df_mapping["topic"] = df_mapping["text_clean"].map(topic_mapping)
    df_mapping = df_mapping.drop(columns=["text_clean"])
    df_mapping.to_parquet(output_dir / "post_topics.parquet", index=False)

    print(f"   -> Salvo em: {output_dir}")

def export_visualizations(topic_model: BERTopic, output_dir: Path):
    """
    Gera e salva os relatórios interativos em HTML.
    """
    print("4. Gerando Visualizações HTML...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df_topic_info = topic_model.get_topic_info()
    n_clusters_total = len(df_topic_info) - 1 # Ignora o tópico -1 (outliers)

    # Gráfico de Barras (Assinaturas Linguísticas)
    fig_words = topic_model.visualize_barchart(
        top_n_topics=n_clusters_total, 
        n_words=10,
        title="Assinaturas Linguísticas do Ecossistema (c-TF-IDF)"
    )
    fig_words.write_html(output_dir / "visualizacao_topicos.html")

    # Mapa Topológico 2D
    fig_2d = topic_model.visualize_topics(
        title="Topologia do Ecossistema: Distância entre Clusters Ideológicos"
    )
    fig_2d.write_html(output_dir / "mapa_2d_topicos.html")

    print("   -> Gerando Árvore Hierárquica...")
    fig_hierarchy = topic_model.visualize_hierarchy(
        custom_labels=True, 
        title="Árvore Hierárquica dos Subreddits"
    )
    fig_hierarchy.write_html(output_dir / "hierarquia_topicos.html")
    
    print(f"   -> Visualizações salvas em: {output_dir}")