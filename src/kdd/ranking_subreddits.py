import re
import pandas as pd

with open("../../reports/relatorio_top_1000_subreddits.md", "r", encoding="utf-8") as f:
    text = f.read()

pattern = (
    r"### \d+º \| (r/[^\n]+)\n"
    r"- \*\*Quantidade:\*\* (\d+) keywords diferentes\n"
    r"- \*\*Keywords:\*\* (.*?)(?=\n### |\Z)"
)

matches = re.findall(pattern, text, re.DOTALL)

rows = []

for subreddit, qtd, keywords in matches:
    kws = [k.strip() for k in keywords.split("•")]

    rows.append({
        "subreddit": subreddit,
        "n_keywords": int(qtd),
        "keywords": kws
    })

df = pd.DataFrame(rows)
df.to_parquet("../../data/processed/ranking_subreddits.parquet")