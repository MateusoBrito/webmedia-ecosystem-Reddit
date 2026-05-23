# 🧠 Pipeline de Topic Modeling (Macro-Comunidades)

Este documento detalha o funcionamento do script de modelagem de tópicos, responsável por agrupar e descobrir a semântica das comunidades (subreddits) utilizando a arquitetura **BERTopic** com inicialização por média de embeddings (Centroides).

## 🗺️ Visão Geral

O script orquestra um pipeline de ponta a ponta: carrega as representações espaciais (centroides) pré-calculadas, alinha-as com os textos concatenados de cada comunidade, reduz a dimensionalidade, clusteriza as comunidades ideologicamente próximas e exporta relatórios analíticos e visuais.

## ⚙️ Etapas do Pipeline

### 1. Preparação e Alinhamento (`prepare_data`)

A base do modelo exige que a matriz matemática de vetores corresponda **exatamente** à ordem dos textos.

* **Carregamento:** Importa a matriz `.parquet` com os vetores médios (centroides) de cada subreddit.
* **Concatenação:** Carrega os textos pré-processados e agrupa todas as postagens de uma mesma comunidade em um único "super-documento" (representação macro).
* **Alinhamento:** Reordena o DataFrame de textos para que o índice 0 seja o mesmo subreddit da linha 0 da matriz NumPy, prevenindo qualquer *mismatch* durante o treinamento.

### 2. Treinamento do Modelo (`train_topic_model`)

Aplica o algoritmo BERTopic customizado para o estado da arte em análise topológica:

* **UMAP (Uniform Manifold Approximation and Projection):** Algoritmo de redução de dimensionalidade
* **HDBSCAN:** Algoritmo de clusterização baseada em densidade. 
* **c-TF-IDF (Class-based TF-IDF):** Avalia a importância das palavras não no nível do post, mas no nível da comunidade, extraindo a "assinatura linguística" de cada cluster.

### 3. Dicionário de Tópicos (`export_topic_dictionary`)

Processa o resultado do agrupamento para criar um mapeamento rastreável de "Quem está onde".

* Cruza o ID do tópico com a lista de subreddits correspondentes.
* Gera um dicionário estruturado contendo o número do tópico, a quantidade de comunidades nele, seu nome, suas principais palavras (representação) e a lista exata de subreddits pertencentes àquele espectro.
* **Saída:** Arquivo leve em formato `.json`.

### 4. Geração de Visualizações (`export_visualizations`)

Cria artefatos HTML interativos para exploração e apresentação dos dados:

* **Gráfico de Barras (Assinaturas):** Exibe as 10 palavras mais representativas (maior score c-TF-IDF) de cada tópico identificado.
* **Mapa 2D (Topologia):** Plota a distância bidimensional entre os clusters, evidenciando proximidades ideológicas e possíveis câmaras de eco.

---

## 💾 Estrutura de Diretórios (Entradas e Saídas)

**Consome:**

* `artifacts/embeddings/subreddit_centroids.parquet` (Matriz de centroides)
* `data/processed/preprocess_text.parquet` (Textos limpos)

**Gera:**

* `reports/topic_modeling/dicionario_topicos.json`
* `reports/topic_modeling/visualizacao_topicos.html`
* `reports/topic_modeling/mapa_2d_topicos.html`

---

## ▶️ Como Executar

O script foi desenhado para ser executado como um módulo a partir da raiz do projeto, garantindo que os caminhos dinâmicos (`ROOT`) sejam resolvidos corretamente.

No terminal, posicionado na raiz do projeto, execute:

```bash
python3 -m src.topic_modeling

```

Aqui está a seção final para você adicionar ao final do seu `README.md`.

Como a modelagem de tópicos (BERTopic/HDBSCAN) é um aprendizado não supervisionado, nós não usamos o tradicional `GridSearchCV` do *scikit-learn* (pois não temos uma variável "Y" para prever). Em vez disso, montamos uma busca customizada avaliando a **qualidade e separação espacial dos clusters**, usando métricas como o *Silhouette Score*.

Basta copiar o bloco abaixo e colar no final do seu arquivo da documentação:

## 🚀 Próximos Passos: Ajuste Fino (Hyperparameter Tuning)

O modelo atual utiliza parâmetros fixos como pontos de partida (ex: `n_neighbors=50` e `min_cluster_size=50`). No entanto, para separarmos os subreddits em clusters bem definidos e separados surge a necessidade de ajustar esses parâmetros, testando diversas combinações.

Avaliar modelos de tópicos (que é um tipo de aprendizado não supervisionado) é um dos maiores desafios da ciência de dados. Como não temos um "gabarito" dizendo qual subreddit pertence a qual grupo, precisamos de métricas matemáticas que avaliem duas coisas: **a geometria dos grupos** (se eles estão bem separados no espaço) e **a qualidade do texto** (se as palavras fazem sentido juntas).

---

### 1. O Conceito de Grid Search (Busca em Grade)

Em aprendizado de máquina, algoritmos como UMAP e HDBSCAN dependem de "botões" que você precisa girar (os hiperparâmetros, como `n_neighbors` ou `min_cluster_size`).

O **Grid Search** é simplesmente a força bruta automatizada para encontrar a sintonia perfeita desses botões.
Você define uma "grade" de opções (ex: testar tamanhos de cluster de 10, 30 e 50). O algoritmo vai criar um modelo para a combinação 1, testar. Criar para a combinação 2, testar... e assim por diante. No final, ele te diz qual combinação obteve a maior nota na sua métrica de avaliação.

A grande pergunta é: *qual métrica usar para dar essa nota?* É aí que entram as três abaixo.

---

### 2. Silhouette Score (Métrica Espacial / Topológica)

O Silhouette Score não lê os textos; ele olha apenas para o mapa 3D (ou 2D) que o UMAP e o HDBSCAN criaram. Ele avalia se os pontos (subreddits) formaram "ilhas" bem definidas.

Para calcular o Silhouette de um ponto, a matemática faz duas perguntas:

1. **Coesão ($a$):** Quão perto este ponto está dos outros pontos do *seu próprio* grupo? (Queremos que seja muito perto).
2. **Separação ($b$):** Quão longe este ponto está do grupo *vizinho mais próximo*? (Queremos que seja muito longe).

A fórmula compara esses dois valores: 

$$s = \frac{b - a}{\max(a, b)}$$

* **Interpretação:** A nota vai de **-1 a 1**.
* **Perto de 1:** Excelente. O subreddit está no centro da sua câmara de eco e muito distante das outras.
* **Perto de 0:** Ruim. O subreddit está exatamente na fronteira entre dois tópicos, indicando sobreposição (grupos muito misturados).
* **Perto de -1:** Péssimo. O subreddit provavelmente foi classificado no tópico errado.



---

### 3. Topic Coherence (Métrica Semântica / de Significado)

Enquanto o Silhouette olha para as distâncias, a **Coerência** olha para as palavras. Ela tenta simular o julgamento humano: *"Esse grupo de palavras faz sentido lógico?"*

Para medir isso, pegamos as 10 palavras mais importantes de um tópico (ex: *bola, campo, gol, juiz*). A matemática então varre um banco de dados gigantesco (pode ser a Wikipedia ou o próprio texto do Reddit) e conta quantas vezes essas palavras aparecem **juntas** na mesma frase ou documento.

* **Interpretação:**
* **Alta Coerência:** Se "bola" e "gol" aparecem juntas o tempo todo na vida real, o modelo acertou em agrupá-las. O tópico é interpretável e focado (ex: Tópico de Futebol).
* **Baixa Coerência:** Se o modelo cria um tópico com as palavras *bola, imposto, júpiter, maquiagem*, essas palavras raramente dividem o mesmo contexto na vida real. O tópico é um "saco de gatos" sem sentido semântico.



---

### 4. Topic Diversity (Métrica de Distinção)

A **Diversidade** mede a redundância do seu modelo global. Ela avalia se os tópicos gerados são realmente diferentes uns dos outros.

A matemática aqui é simples: pegamos as principais palavras de *todos* os tópicos criados e calculamos a porcentagem de palavras que são únicas.

* **Interpretação:** A nota vai de **0 a 1** (ou 0% a 100%).
* **Alta Diversidade (perto de 1):** Significa que cada câmara de eco do Reddit tem seu próprio vocabulário exclusivo. O modelo conseguiu mapear nichos bem distintos.
* **Baixa Diversidade (perto de 0):** O modelo fracassou por "espalhamento". Por exemplo, ele criou o Tópico 1 (*futebol, bola, gol*) e o Tópico 2 (*bola, gol, juiz*). Na prática, esses dois tópicos deveriam ser um só. Baixa diversidade significa que o modelo dividiu demais o que não precisava ser dividido.