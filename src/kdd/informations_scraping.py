from src.utils.data_loader import load_preprocessed_data
import pandas as pd

# Carrega dados
df = load_preprocessed_data()

# --------------------------------------------------
# Garantir formato de data
# --------------------------------------------------
#df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

# --------------------------------------------------
# 1. Quantidade total de posts
# --------------------------------------------------
total_posts = len(df)

# --------------------------------------------------
# 2. Quantidade de subreddits únicos
# --------------------------------------------------
num_subreddits = df["subreddit"].nunique()

# --------------------------------------------------
# 3. Período analisado
# --------------------------------------------------
#data_inicio = df["timestamp"].min()
#data_fim = df["timestamp"].max()

# --------------------------------------------------
# Resultado
# --------------------------------------------------
print("=" * 50)
print("ESTATÍSTICAS DO DATASET")
print("=" * 50)

print(f"Total de posts: {total_posts:,}")
print(f"Total de subreddits: {num_subreddits:,}")
#print(f"Período analisado: {data_inicio} → {data_fim}")