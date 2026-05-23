#!/bin/bash

set -e

# Definição de variáveis
EMBEDDINGS_PATH=${1:-"artifacts/embeddings/embeddings_paraphrase.npz"}
EMBEDDINGS_MODEL="paraphrase-multilingual-MiniLM-L12-v2"
CENTROIDS_PATH=${2:-"artifacts/embeddings/subreddit_centroids.parquet"}

echo "======================================"
echo "Iniciando Pipeline de Representação"
echo "======================================"
echo "Modelo: $EMBEDDINGS_MODEL"
echo "Embeddings: $EMBEDDINGS_PATH"
echo "Centroids: $CENTROIDS_PATH"
echo "--------------------------------------"

# Passo 1: Gerar embeddings dos posts (Codificação Independente)
echo "[1/2] Gerando Embeddings..."
python -m src.generate_embeddings \
  --output "$EMBEDDINGS_PATH" \
  --model "$EMBEDDINGS_MODEL"

# Passo 2: Calcular centroides (Agregação Inter-documentos)
# NOTA: Removido o .py do final do módulo
echo "[2/2] Calculando Centroides das Comunidades..."
python -m src.calculate_subreddit_centroids \
  --embeddings "$EMBEDDINGS_PATH" \
  --output "$CENTROIDS_PATH"

echo "======================================"
echo "Pipeline concluído com sucesso!"
echo "======================================"