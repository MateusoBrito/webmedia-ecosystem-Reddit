"""Grid search exaustivo de hiperparametros UMAP+HDBSCAN para BERTopic.

Equivalente Python adaptado do notebook externo `14_TM_parameters.ipynb`
(ver external_code/01_grid_search_TM_parameters/README.md).

Pipeline:
    1. Le parquet master pos-processado, filtra (text_processed >= 3 tokens),
       amostra `--sample-frac` (default 1%).
    2. Para cada embedding model em `--embedding-models`, encoda os docs
       (com cache em outputs/topics/_cache/__gpu.npy quando disponivel).
    3. Itera sobre o grid de hiperparametros, treinando BERTopic em cada
       combinacao e calculando 3 metricas via src.tm_metrics.
    4. Salva resultado consolidado em `outputs/topics/grid_search/<study>.csv`.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import random
import time
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from tqdm.auto import tqdm

from src.tm_metrics import compute_all_metrics

console = Console()

# --------------------------------------------------------------------------- #
# Constantes — paths e seeds
# --------------------------------------------------------------------------- #
PARQUET = Path("data/processed/ads_deputados_preprocessed.parquet")
CACHE_DIR = Path("outputs/topics/_cache")
OUTPUT_DIR = Path("outputs/topics/grid_search")
SEED = 42

# --------------------------------------------------------------------------- #
# Presets de espaco de busca (cf docs/pipeline_grid_search.md §3, decisoes 2+3)
# --------------------------------------------------------------------------- #
PRESETS: dict[str, dict] = {

    "notebook": dict(
        n_neighbors=[10, 20],
        n_components=[5, 10],
        min_dist=[0.0, 0.1, 0.5],
        min_cluster_size=[50, 100, 150],
        min_samples=[5, 10, 100],
        cluster_selection_epsilon=[0.1, 0.5],
        sample_frac=0.01,
    ),

    "D": dict(
        n_neighbors=[10, 20],
        n_components=[5, 10],
        min_dist=[0.0, 0.1, 0.5],
        min_cluster_size=[10, 20, 50],
        min_samples=[3, 5, 10],
        cluster_selection_epsilon=[0.1, 0.5],
        sample_frac=0.01,
    ),
    "E": dict(
        n_neighbors=[10, 15, 20],
        n_components=[5, 10],
        min_dist=[0.0, 0.1],
        min_cluster_size=[50, 100, 150],
        min_samples=[5, 10],
        cluster_selection_epsilon=[0.0, 0.5],
        sample_frac=0.10,
    ),

    "G1_borda": dict(
        n_neighbors=[10, 15],
        n_components=[5, 10],
        min_dist=[0.0],
        min_cluster_size=[25, 30, 50],
        min_samples=[3, 5],
        cluster_selection_epsilon=[0.0],
        sample_frac=0.10,
    ),

    # H_gpu_full: grid no corpus completo (266k) com UMAP+HDBSCAN em GPU (cuml).
    # min_cluster_size escalado para 266k (50 daria ~3000 microtopicos).
    # 192 combos × ~2-4min/combo na 5070Ti = ~6-13h.
    "H_gpu_full": dict(
        n_neighbors=[10, 15, 30],
        n_components=[5, 10],
        min_dist=[0.0, 0.1],
        min_cluster_size=[100, 250, 500, 1000],
        min_samples=[5, 10],
        cluster_selection_epsilon=[0.0, 0.2],
        sample_frac=1.0,
    ),

    # ----------------------------------------------------------------------- #
    # Pivot 2026-05-04 — apos H_gpu_full_det_v1 mostrar que top combos tinham
    # noise 56-68% e silhouette mediana negativa. Decisao: rodar A + B em
    # sequencia, totalizando 360 combos (~3.5-4h GPU).
    # ----------------------------------------------------------------------- #

    # H_v2_low_noise: reduzir noise pra <40%. Aposta: mcs/min_samples menores
    # + n_components maior (mais espaco UMAP -> silhouette positivo).
    # 3×3×2×4×2×1 = 144 combos. ~1.5h GPU.
    "H_v2_low_noise": dict(
        n_neighbors=[15, 30, 50],
        n_components=[10, 15, 20],
        min_dist=[0.0, 0.1],
        min_cluster_size=[50, 100, 150, 200],
        min_samples=[3, 5],
        cluster_selection_epsilon=[0.0],
        sample_frac=1.0,
    ),

    # H_v2_around_winner: refinar ao redor do top combo do det_v1
    # (mcs=1000, ms=10, n_nei=30, n_comp=10, mdist=0.1 -> n=12, sil=+0.56).
    "H_v2_around_winner": dict(
        n_neighbors=[20, 30, 40],
        n_components=[10, 15],
        min_dist=[0.0, 0.1, 0.2],
        min_cluster_size=[500, 750, 1000, 1500],
        min_samples=[5, 10, 15],
        cluster_selection_epsilon=[0.0],
        sample_frac=1.0,
    ),

    # ----------------------------------------------------------------------- #
    # H_v3_sweet_spot (2026-05-08) — refinar regiao vencedora do v2_low_noise.
    # Os 11 trials saudaveis do A todos tinham ms=3 e mcs=50-150.
    # ----------------------------------------------------------------------- #
    "H_v3_sweet_spot": dict(
        n_neighbors=[10, 15, 20, 25],
        n_components=[10, 15, 20, 25],
        min_dist=[0.0],
        min_cluster_size=[50, 75, 100, 150, 200],
        min_samples=[3, 5],
        cluster_selection_epsilon=[0.0],
        sample_frac=1.0,
    ),

    # ----------------------------------------------------------------------- #
    # H_v4_refine_high_sil (2026-05-14) — refinamento focado pos-experimento.
    # Vencedor do experimento (B1×MiniLM high_sil) tinha: n_nei=25, n_comp=25,
    # mcs=200, ms=3. Treino full corpus revelou topicos contaminados:
    #   - T4 "pompeo matto, pompeo trabalhar" (nome proprio dominante)
    #   - T9 "whatsapp, educacao milhao, construcao creche" (marketing fragmentado)
    # Hipotese: aumentar mcs e min_samples elimina campanhas pessoais pequenas
    # (centenas de anuncios do mesmo deputado) sem perder temas reais.
    # Fixa n_nei=25, n_comp=25, min_dist=0.0 (ja validados). Varia so mcs e ms.
    # 1×1×1×6×5×1 = 30 combos. ~20min GPU.
    # ----------------------------------------------------------------------- #
    "H_v4_refine_high_sil": dict(
        n_neighbors=[25],
        n_components=[25],
        min_dist=[0.0],
        min_cluster_size=[200, 300, 400, 500, 750, 1000],
        min_samples=[3, 5, 7, 10, 15],
        cluster_selection_epsilon=[0.0],
        sample_frac=1.0,
    ),
}

# Embedding models (decisao 1)
DEFAULT_EMBEDDING_MODELS = ["paraphrase-multilingual-mpnet-base-v2"]
ALL_EMBEDDING_MODELS = [
    "paraphrase-multilingual-mpnet-base-v2",
    "paraphrase-multilingual-MiniLM-L12-v2",
    "intfloat/multilingual-e5-large",
]

# BERTopic / Vectorizer (decisao 5: TfidfVectorizer)
TOP_N_WORDS = 10  # consistente com tm_metrics e nb 15
NGRAM_RANGE = (1, 2)


# --------------------------------------------------------------------------- #
# Utilitarios
# --------------------------------------------------------------------------- #
def set_deterministic(seed: int = SEED) -> None:
    """Aproxima determinismo em CPU/GPU antes de importar/usar torch.

    Para embeddings em GPU, `CUBLAS_WORKSPACE_CONFIG` precisa existir antes
    do primeiro import de torch no processo. Como este script importa torch
    dentro de `encode_docs`, chamar esta funcao no inicio do `main` ainda e
    cedo o bastante.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    except ImportError:
        pass


def _corpus_fingerprint(texts: list[str]) -> str:
    """Mesma logica de embed_gpu.corpus_fingerprint / topic_modeling._corpus_fingerprint."""
    h = hashlib.sha256()
    h.update(str(len(texts)).encode())
    for i in (0, len(texts) // 4, len(texts) // 2, 3 * len(texts) // 4, len(texts) - 1):
        if 0 <= i < len(texts):
            h.update(texts[i].encode("utf-8", errors="ignore")[:512])
    return h.hexdigest()[:16]


def _safe_model_name(name: str) -> str:
    return name.replace("/", "_")


def load_sample(
    sample_frac: float,
    text_column_embed: str = "text_clean",
    text_column_vec: str = "text_processed",
    dedup: bool = True,
) -> tuple[list[str], list[str], str]:
    """Le parquet, filtra, amostra. Retorna (docs_embed, docs_vec, fingerprint).

    Pipeline hibrido (decisao 2026-05-03):
        - text_column_embed: usado pra embeddings + fit (default text_clean,
          natural — mpnet performa melhor em texto natural)
        - text_column_vec: usado pelo TfidfVectorizer + c_v (default
          text_processed, lematizado e sem stopwords — top-words ficam reais
          em vez de "que/de/para/com")

    Dedup eh feita por text_column_embed (a coluna que de fato vai pro fit).
    Fingerprint eh calculado sobre docs_embed (consistente com cache de
    embeddings pre-existente).

    Args:
        dedup: se True (default), aplica drop_duplicates por text_column_embed
            apos o filtro de >=3 tokens. Se False, mantem todas as duplicatas
            (corpus completo com re-impulsionamentos — experimento 2026-05-10).
    """
    if not PARQUET.exists():
        raise FileNotFoundError(f"Parquet master nao existe: {PARQUET}")
    df = pd.read_parquet(PARQUET)
    df = df[df["text_processed"].fillna("").str.split().str.len() >= 3].copy()
    df = df.reset_index(drop=True)
    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=SEED).reset_index(drop=True)

    if dedup:
        n_before = len(df)
        df = df.drop_duplicates(subset=[text_column_embed]).reset_index(drop=True)
        n_dropped = n_before - len(df)
        if n_dropped > 0:
            console.print(f"[dim]Dedup: removidas {n_dropped} duplicatas em {text_column_embed}[/dim]")
    else:
        console.print(f"[yellow]NO-DEDUP[/yellow]: mantendo {len(df):,} docs (com duplicatas)")

    docs_embed = df[text_column_embed].fillna("").astype(str).tolist()
    docs_vec = df[text_column_vec].fillna("").astype(str).tolist()
    fp = _corpus_fingerprint(docs_embed)
    return docs_embed, docs_vec, fp


def encode_docs(
    model_name: str,
    docs: list[str],
    fingerprint: str,
    require_gpu: bool = False,
    use_cache: bool = False,
) -> np.ndarray:
    """Encoda docs, opcionalmente usando cache.

    Cache name segue convencao de embed_gpu.py:
    `<safe_model>__n<N>__seed<SEED>__<fp>__gpu.npy`.

    Por padrao o grid nao usa cache. Isso permite validar se a etapa de
    embeddings em GPU tambem esta reprodutivel, em vez de apenas reutilizar
    uma matriz `.npy` ja congelada.
    """
    safe = _safe_model_name(model_name)
    cache_path = CACHE_DIR / f"{safe}__n{len(docs)}__seed{SEED}__{fingerprint}__gpu.npy"
    fp_path = cache_path.with_suffix(".fingerprint")

    if use_cache and cache_path.exists() and fp_path.exists() and fp_path.read_text().strip() == fingerprint:
        console.print(f"[green]CACHE HIT[/green] {cache_path.name}")
        return np.load(cache_path)

    import torch
    from sentence_transformers import SentenceTransformer

    set_deterministic(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if require_gpu and device != "cuda":
        raise RuntimeError(
            "Execucao com --gpu exige CUDA tambem para embeddings, "
            "mas torch.cuda.is_available() retornou False."
        )

    console.print(f"[cyan]Encoding[/cyan] {len(docs):,} docs com {model_name} em {device}")
    sbert = SentenceTransformer(model_name, device=device)
    sbert.eval()
    t0 = time.time()
    with torch.inference_mode():
        emb = sbert.encode(
            docs, batch_size=64, show_progress_bar=False,
            convert_to_numpy=True, normalize_embeddings=True,
        ).astype(np.float32)
    console.print(f"  encoded em {time.time() - t0:.1f}s shape={emb.shape}")

    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, emb)
        fp_path.write_text(fingerprint)
    del sbert
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return emb


def build_bertopic(embeddings_array: np.ndarray, params: dict, use_gpu: bool = False):
    """Constroi BERTopic com params do trial. Embeddings sao passados
    pre-computados no fit_transform — embedding_model do BERTopic e None
    pra evitar re-encode.

    Args:
        use_gpu: se True, usa cuml.UMAP + cuml.HDBSCAN (GPU). Caso contrario
            usa umap-learn + hdbscan (CPU). GPU eh ~10-30x mais rapido em
            corpora grandes. No caminho GPU, o UMAP recebe random_state fixo;
            o transform_seed existe no umap-learn CPU, mas nao no cuML.
    """
    from bertopic import BERTopic
    from bertopic.vectorizers import ClassTfidfTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer

    if use_gpu:
        from cuml import UMAP as cuUMAP
        from cuml.cluster import HDBSCAN as cuHDBSCAN
        umap_model = cuUMAP(
            n_neighbors=params["n_neighbors"],
            n_components=params["n_components"],
            min_dist=params["min_dist"],
            metric="cosine",
            random_state=SEED,
            verbose=False,
        )
        hdbscan_model = cuHDBSCAN(
            min_cluster_size=params["min_cluster_size"],
            min_samples=params["min_samples"],
            cluster_selection_epsilon=params["cluster_selection_epsilon"],
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
            gen_min_span_tree=True,
        )
    else:
        from umap import UMAP
        from hdbscan import HDBSCAN
        umap_model = UMAP(
            n_neighbors=params["n_neighbors"],
            n_components=params["n_components"],
            min_dist=params["min_dist"],
            metric="cosine",
            random_state=SEED,
            transform_seed=SEED,
            verbose=False,
        )
        hdbscan_model = HDBSCAN(
            min_cluster_size=params["min_cluster_size"],
            min_samples=params["min_samples"],
            cluster_selection_epsilon=params["cluster_selection_epsilon"],
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
            core_dist_n_jobs=1,
        )

    vectorizer = TfidfVectorizer(stop_words=None, ngram_range=NGRAM_RANGE)
    ctfidf = ClassTfidfTransformer(reduce_frequent_words=True)

    return BERTopic(
        language="multilingual",
        top_n_words=TOP_N_WORDS,
        embedding_model=None,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        ctfidf_model=ctfidf,
        verbose=False,
    )


def run_one_combo(
    docs_embed: list[str],
    docs_vec: list[str],
    embeddings: np.ndarray,
    params: dict,
    use_gpu: bool = False,
    silhouette_sample_size: int | None = 10_000,
) -> dict:
    """Treina 1 BERTopic (hibrido), calcula metricas, retorna dict.

    Pipeline hibrido:
        1. fit_transform(docs_embed) — UMAP+HDBSCAN sobre embeddings de
           docs_embed (text_clean por default)
        2. update_topics(docs_vec) — recalcula c-TF-IDF sobre docs_vec
           (text_processed por default), produzindo top-words com
           lematizacao e sem stopwords
        3. c_v eh computado sobre docs_vec (alinhado com vocab dos top-words)

    Retorna sempre dict valido (mesmo em erro) com `error` field preenchido.
    """
    out = dict(params)
    try:
        set_deterministic(SEED)
        model = build_bertopic(embeddings, params, use_gpu=use_gpu)
        topics, _ = model.fit_transform(docs_embed, embeddings=embeddings)
        labels = np.asarray(topics)

        # Hibrido: recalcula c-TF-IDF sobre docs_vec (lematizado/sem stopwords)
        if docs_vec is not docs_embed:
            model.update_topics(docs_vec, topics=labels.tolist())

        raw = model.get_topics()
        topics_words = [
            [w for w, _ in raw[t][:TOP_N_WORDS]]
            for t in raw if t != -1
        ]

        # cuml.UMAP guarda embedding_ tambem; se falhar, transform
        reduced = getattr(model.umap_model, "embedding_", None)
        if reduced is None or reduced.shape[1] != params["n_components"]:
            reduced = model.umap_model.transform(embeddings)
        # cuml retorna cuDF/cupy as vezes — converte
        if hasattr(reduced, "to_numpy"):
            reduced = reduced.to_numpy()
        elif hasattr(reduced, "get"):
            reduced = reduced.get()
        reduced = np.asarray(reduced)

        metrics = compute_all_metrics(
            topics_words, docs_vec, reduced, labels,
            silhouette_sample_size=silhouette_sample_size,
        )
        out.update(metrics)
        out["noise_pct"] = float((labels == -1).sum() / len(labels) * 100)
        out["error"] = None
    except Exception as e:
        out.update(dict(c_v=0.0, diversity=0.0, silhouette=0.0,
                        mean_score=0.0, mean_cv_sil=0.0, n_topics=0,
                        noise_pct=100.0))
        out["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    set_deterministic(SEED)

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--study", type=str, required=True,
                        help="Nome do estudo (vira nome do CSV de saida)")
    parser.add_argument("--preset", type=str, default="notebook",
                        choices=list(PRESETS.keys()),
                        help="Preset do espaco de busca + sample_frac. "
                             "Default 'notebook' (cf docstring).")
    parser.add_argument("--sample-frac", type=float, default=None,
                        help="Override do sample_frac do preset")
    parser.add_argument("--text-column", type=str, default="text_clean",
                        choices=["text_clean", "text_processed"],
                        help="Coluna usada para EMBEDDINGS+fit (default text_clean)")
    parser.add_argument("--vectorizer-text-column", type=str, default="text_processed",
                        choices=["text_clean", "text_processed"],
                        help="Coluna usada pelo TfidfVectorizer + c_v "
                             "(default text_processed — lematizado/sem stopwords). "
                             "Igual a --text-column desativa o pipeline hibrido.")
    parser.add_argument("--gpu", action="store_true",
                        help="Usa cuml.UMAP + cuml.HDBSCAN (GPU, ~10-30x mais rapido). "
                             "Requer ambiente RAPIDS (ex: WSL rapids-25). "
                             "Usa seeds fixas, mas cuML pode nao ser bit-a-bit deterministico.")
    parser.add_argument("--silhouette-sample", type=int, default=10_000,
                        help="Sample size pra silhouette (default 10000). 0 = todos os docs (LENTO).")
    parser.add_argument("--resume", action="store_true",
                        help="Se CSV de saida ja existir, pula combos ja feitos.")
    parser.add_argument("--no-dedup", action="store_true",
                        help="Desativa drop_duplicates por text_column. Default eh dedup ativo "
                             "(experimento 2026-05-10: testa pipeline com corpus completo de "
                             "duplicatas/re-impulsionamentos).")
    parser.add_argument("--output-base", type=Path, default=Path("outputs/topics"),
                        help="Diretorio base para outputs. CSVs do grid vao pra "
                             "<output-base>/grid_search/<study>_trials.csv. "
                             "Default: outputs/topics (caminho A1). "
                             "Outros caminhos: outputs/topics/A2_hibrido_nodedup, "
                             "outputs/topics/B1_processed_dedup, outputs/topics/B2_processed_nodedup.")
    parser.add_argument("--embedding-cache", action="store_true",
                        help="Reusa/salva cache de embeddings. Por padrao fica desligado "
                             "para validar a reprodutibilidade do encode em GPU.")
    parser.add_argument("--top-n-words", type=int, default=TOP_N_WORDS,
                        help=f"Top-N words por topico para metricas (default {TOP_N_WORDS})")
    # Overrides individuais do espaco — se None, usa do preset
    parser.add_argument("--n-neighbors", type=int, nargs="+", default=None)
    parser.add_argument("--n-components", type=int, nargs="+", default=None)
    parser.add_argument("--min-dist", type=float, nargs="+", default=None)
    parser.add_argument("--min-cluster-size", type=int, nargs="+", default=None)
    parser.add_argument("--min-samples", type=int, nargs="+", default=None)
    parser.add_argument("--cluster-selection-epsilon", type=float, nargs="+", default=None)
    embed_group = parser.add_mutually_exclusive_group()
    embed_group.add_argument("--all-embeddings", action="store_true",
                             help="Usa os 3 embedding models (mpnet + minilm-L12 + e5-large)")
    embed_group.add_argument("--embedding-models", nargs="+", default=None,
                             help="Lista de embedding models customizada")
    args = parser.parse_args()

    # Resolve preset + overrides
    space_cfg = dict(PRESETS[args.preset])
    if args.sample_frac is not None:
        space_cfg["sample_frac"] = args.sample_frac
    overrides = {
        "n_neighbors": args.n_neighbors,
        "n_components": args.n_components,
        "min_dist": args.min_dist,
        "min_cluster_size": args.min_cluster_size,
        "min_samples": args.min_samples,
        "cluster_selection_epsilon": args.cluster_selection_epsilon,
    }
    for k, v in overrides.items():
        if v is not None:
            space_cfg[k] = v

    if args.all_embeddings:
        embedding_models = ALL_EMBEDDING_MODELS
    elif args.embedding_models:
        embedding_models = args.embedding_models
    else:
        embedding_models = DEFAULT_EMBEDDING_MODELS

    sample_frac = space_cfg["sample_frac"]

    # Calculo do espaco de busca
    n_combos = (
        len(space_cfg["n_neighbors"]) * len(space_cfg["n_components"])
        * len(space_cfg["min_dist"]) * len(space_cfg["min_cluster_size"])
        * len(space_cfg["min_samples"]) * len(space_cfg["cluster_selection_epsilon"])
        * len(embedding_models)
    )

    hybrid = args.text_column != args.vectorizer_text_column
    output_dir = args.output_base / "grid_search"
    console.print(Panel.fit(
        f"[bold]Grid Search BERTopic — caminho Lucas[/bold]\n"
        f"Estudo: [cyan]{args.study}[/cyan]  Preset: [cyan]{args.preset}[/cyan]\n"
        f"Sample: [cyan]{sample_frac * 100:.1f}%[/cyan] do corpus master\n"
        f"Output base: [cyan]{args.output_base}[/cyan]\n"
        f"Embed col: [cyan]{args.text_column}[/cyan]  "
        f"Vec col: [cyan]{args.vectorizer_text_column}[/cyan]  "
        f"Hibrido: [cyan]{hybrid}[/cyan]\n"
        f"Dedup: [cyan]{not args.no_dedup}[/cyan]  "
        f"Backend: [cyan]{'GPU (cuml)' if args.gpu else 'CPU (umap-learn+hdbscan)'}[/cyan]  "
        f"Resume: [cyan]{args.resume}[/cyan]  "
        f"Emb cache: [cyan]{args.embedding_cache}[/cyan]\n"
        f"Embedding models ({len(embedding_models)}): "
        + ", ".join(embedding_models)
        + f"\n  n_neighbors={space_cfg['n_neighbors']}"
        + f"  n_components={space_cfg['n_components']}"
        + f"  min_dist={space_cfg['min_dist']}\n"
        + f"  mcs={space_cfg['min_cluster_size']}"
        + f"  min_samples={space_cfg['min_samples']}"
        + f"  cluster_eps={space_cfg['cluster_selection_epsilon']}"
        + f"\n[bold]Total combos:[/bold] [yellow]{n_combos}[/yellow]",
        border_style="magenta",
    ))

    # 1. Carrega sample (hibrido: 2 colunas)
    docs_embed, docs_vec, fp = load_sample(
        sample_frac,
        text_column_embed=args.text_column,
        text_column_vec=args.vectorizer_text_column,
        dedup=not args.no_dedup,
    )
    console.print(f"Docs: [bold]{len(docs_embed):,}[/bold] | fingerprint: {fp}\n")

    # 2. Encoda cada embedding model (sempre sobre docs_embed)
    embeddings_by_model: dict[str, np.ndarray] = {}
    for em in embedding_models:
        embeddings_by_model[em] = encode_docs(
            em,
            docs_embed,
            fp,
            require_gpu=args.gpu,
            use_cache=args.embedding_cache,
        )

    # 3. Checkpoint: se CSV existe e --resume, pula combos ja feitos
    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / f"{args.study}_trials.csv"
    done_keys: set[tuple] = set()
    if args.resume and out_csv.exists():
        df_existing = pd.read_csv(out_csv)
        for _, r in df_existing.iterrows():
            done_keys.add((
                int(r["n_neighbors"]), int(r["n_components"]), float(r["min_dist"]),
                int(r["min_cluster_size"]), int(r["min_samples"]),
                float(r["cluster_selection_epsilon"]), str(r["embedding_model"]),
            ))
        console.print(f"[yellow]RESUME[/yellow]: {len(done_keys)} combos ja em {out_csv.name}, pulando\n")

    # 4. Loop do grid — append por combo (crash-safe)
    sil_sample = args.silhouette_sample if args.silhouette_sample > 0 else None
    t_start = time.time()
    space = list(product(
        space_cfg["n_neighbors"], space_cfg["n_components"], space_cfg["min_dist"],
        space_cfg["min_cluster_size"], space_cfg["min_samples"],
        space_cfg["cluster_selection_epsilon"],
        embedding_models,
    ))

    for i, (n_nei, n_comp, mdist, mcs, ms, ceps, em_name) in enumerate(
        tqdm(space, desc="Grid", unit="combo"), start=1
    ):
        key = (n_nei, n_comp, mdist, mcs, ms, ceps, em_name)
        if key in done_keys:
            continue

        params = dict(
            n_neighbors=n_nei, n_components=n_comp, min_dist=mdist,
            min_cluster_size=mcs, min_samples=ms,
            cluster_selection_epsilon=ceps,
            embedding_model=em_name,
        )
        t0 = time.time()
        out = run_one_combo(
            docs_embed, docs_vec, embeddings_by_model[em_name],
            params, use_gpu=args.gpu, silhouette_sample_size=sil_sample,
        )
        out["elapsed_s"] = round(time.time() - t0, 1)
        out["trial"] = i

        # Append imediato (crash-safe)
        df_row = pd.DataFrame([out])
        write_header = not out_csv.exists()
        df_row.to_csv(out_csv, index=False, encoding="utf-8-sig",
                      mode="a", header=write_header)

        if out["error"]:
            console.print(f"[red]ERR[/red] trial {i}/{n_combos}: {out['error']}")
        else:
            console.print(
                f"trial {i:3d}/{n_combos}  "
                f"{em_name.split('/')[-1][:30]:30s}  "
                f"N={out['n_topics']:3d}  "
                f"noise={out['noise_pct']:5.1f}%  "
                f"c_v={out['c_v']:+.3f}  div={out['diversity']:.3f}  sil={out['silhouette']:+.3f}  "
                f"mean={out['mean_score']:+.4f}  cv_sil={out['mean_cv_sil']:+.4f}  ({out['elapsed_s']:.0f}s)"
            )

    elapsed_total = time.time() - t_start
    console.rule(f"[bold green]Grid concluido em {elapsed_total / 60:.1f} min[/bold green]")
    df_results = pd.read_csv(out_csv)
    console.print(f"\n[green]OK[/green] {out_csv}  ({len(df_results)} trials totais)")

    # 5. Top 10 por mean_score
    top = df_results.dropna(subset=["mean_score"]).nlargest(10, "mean_score")
    cols = ["trial", "embedding_model", "n_neighbors", "n_components", "min_dist",
            "min_cluster_size", "min_samples", "cluster_selection_epsilon",
            "n_topics", "noise_pct", "c_v", "diversity", "silhouette", "mean_score"]
    t = Table(title="Top 10 trials (por mean_score)", show_lines=False)
    for col in cols:
        t.add_column(col, justify="right")
    for _, row in top.iterrows():
        t.add_row(*[
            f"{row[c]:.3f}" if isinstance(row[c], float) else str(row[c])[:25]
            for c in cols
        ])
    console.print(t)

    # 6. Resumo final
    n_failed = df_results["error"].notna().sum()
    console.print(f"\nFalhas: {n_failed}/{len(df_results)}")
    if n_failed > 0:
        console.print(f"[yellow]Trials com erro estao no CSV (campo `error`)[/yellow]")


if __name__ == "__main__":
    main()
