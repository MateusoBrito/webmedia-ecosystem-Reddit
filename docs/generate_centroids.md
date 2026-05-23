# Pipeline de Embeddings e Centroides

Este documento descreve o funcionamento e a execução dos dois principais scripts do pipeline de representação vetorial de subreddits:

1. `generate_embeddings.py` (Fase 1: Codificação Independente)
2. `calculate_subreddit_centroids.py` (Fase 2: Agregação Inter-Documentos)

---

## Visão Geral do Pipeline

O pipeline processa os dados textuais seguindo a arquitetura de inicialização por média:

**Textos Limpos (Posts)** $\rightarrow$ **Embeddings Isolados (Vetores Semânticos)** $\rightarrow$ **Centroides (Macro-representação do Subreddit)**

---

## 1. Gerador de Embeddings (`generate_embeddings.py`)

### Objetivo

Transformar textos de posts em representações vetoriais (embeddings) no nível do documento, usando modelos da biblioteca `sentence-transformers`. Cada post é convertido em um vetor numérico que captura seu significado semântico espacial (Mean Pooling dos tokens).

### ⚙️ O que o script faz

* Carrega os dados pré-processados.
* Seleciona a coluna `text_clean`.
* Gera embeddings para cada texto de forma independente utilizando inferência em GPU (se disponível).
* Associa cada embedding ao `id` do seu respectivo post.
* Salva as matrizes otimizadas em formato `.npz`.

### 🧠 Modelo Utilizado

Por padrão: `paraphrase-multilingual-MiniLM-L12-v2`

* **Vantagens:** Suporta múltiplos idiomas de forma nativa, é computacionalmente leve e gera vetores robustos de 384 dimensões.

### 💾 Saída

Gera o arquivo de matrizes comprimidas (ex: `artifacts/embeddings/embeddings_posts.npz`) contendo:

* `embeddings`: Matriz NumPY de formato `(N_posts, 384)`.
* `ids`: Array com os identificadores únicos dos posts correspondentes.

### ▶️ Como rodar

Execute o script passando o caminho de saída e, opcionalmente, o modelo e o tamanho do lote (*batch size*):

```bash
python -m src.generate_embeddings \
  --output artifacts/embeddings/embeddings_posts.npz \
  --model paraphrase-multilingual-MiniLM-L12-v2 \
  --batch_size 128

```

**Parâmetros:**

* `--output` (Obrigatório): Caminho onde o arquivo `.npz` será salvo.
* `--model` (Opcional): Nome do modelo no HuggingFace (padrão: `paraphrase-multilingual-MiniLM-L12-v2`).
* `--batch_size` (Opcional): Tamanho do lote para processamento (padrão: `128`). Aumente se tiver mais VRAM disponível na GPU.

---

## 2. Calculadora de Centroides (`calculate_subreddit_centroids.py`)

### 🎯 Objetivo

Criar a representação semântica macro de cada comunidade (Subreddit). O script agrupa os embeddings isolados gerados na etapa anterior e calcula o centroide matemático do grupo.

### ⚙️ O que o script faz

* Carrega o DataFrame limpo original e a matriz de embeddings `.npz`.
* Valida a correspondência exata (match) entre as linhas do DataFrame e a matriz.
* Agrupa os índices dos posts por `subreddit`.
* Calcula a média aritmética geométrica (`mean(axis=0)`) de todos os vetores pertencentes àquela comunidade.
* Salva o resultado em um formato colunar leve.

### 💾 Saída

Gera um arquivo `.parquet` contendo:

* **Índice:** Nome do `subreddit`.
* **Colunas:** As 384 dimensões representando o vetor médio da comunidade.

### ▶️ Como rodar

Execute o script apontando para o `.npz` gerado anteriormente e definindo onde o arquivo final das comunidades será salvo:

```bash
python -m src.calculate_subreddit_centroids \
  --embeddings artifacts/embeddings/embeddings_posts.npz \
  --output artifacts/embeddings/centroids_subreddits.parquet

```

**Parâmetros:**

* `--embeddings` (Obrigatório): Caminho para o arquivo `.npz` gerado no Passo 1.
* `--output` (Obrigatório): Caminho onde o DataFrame `.parquet` com os centroides será salvo.

---

## 🚀 Executando o Pipeline Completo

Para rodar todo o processo de ponta a ponta de forma sequencial, você pode encadear os comandos no seu terminal:

```bash
# 1. Gera os vetores dos posts isolados
python -m src.generate_embeddings --output artifacts/embeddings/embeddings_posts.npz

# 2. Agrega os vetores para formar as comunidades
python -m src.calculate_subreddit_centroids --embeddings artifacts/embeddings/embeddings_posts.npz --output artifacts/embeddings/centroids_subreddits.parquet

```