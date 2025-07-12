from pathlib import Path
import nbformat as nbf

# Paths
notebook_dir = Path("notebooks")
notebook_dir.mkdir(parents=True, exist_ok=True)
notebook_path = notebook_dir / "evaluate_models_from_db.ipynb"

nb = nbf.v4.new_notebook()
cells = []

# 📌 Header
cells.append(nbf.v4.new_markdown_cell("# 📊 Evaluation of IR Models (From Database)"))

# 📦 Imports and metric functions
cells.append(nbf.v4.new_code_cell("""
from sqlalchemy.orm import Session
from database import SessionLocal
from repositories import query_repo, qrel_repo, result_repo
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Metrics
def precision_at_k(ranked_docs, relevant_docs, k):
    return len(set(ranked_docs[:k]) & set(relevant_docs)) / k

def recall_at_k(ranked_docs, relevant_docs, k):
    return len(set(ranked_docs[:k]) & set(relevant_docs)) / len(relevant_docs) if relevant_docs else 0

def average_precision(ranked_docs, relevant_docs):
    hits = 0
    sum_precisions = 0.0
    for i, doc_id in enumerate(ranked_docs):
        if doc_id in relevant_docs:
            hits += 1
            sum_precisions += hits / (i + 1)
    return sum_precisions / len(relevant_docs) if relevant_docs else 0

def ndcg_at_k(ranked_docs, relevant_docs, k):
    def dcg(rel):
        return sum((2**r - 1) / np.log2(i + 2) for i, r in enumerate(rel))
    rel = [1 if doc_id in relevant_docs else 0 for doc_id in ranked_docs[:k]]
    ideal_rel = sorted(rel, reverse=True)
    return dcg(rel) / (dcg(ideal_rel) + 1e-10)

db: Session = SessionLocal()
qrels = defaultdict(set)
for qrel in qrel_repo.get_all(db):
    qrels[qrel.query_id].add(qrel.doc_id)
print(f"✅ Loaded {len(qrels)} qrels")
"""))

# Algorithms
algorithms = ["vsm", "word2vec", "bm25", "hybrid"]
for algo in algorithms:
    display_name = algo.upper()
    
    cells.append(nbf.v4.new_markdown_cell(f"## 🔎 Evaluation for **{display_name}**"))

    cells.append(nbf.v4.new_code_cell(f"""
# --- {display_name} Evaluation ---
k = 10
results = result_repo.get_results_by_algorithm(db, "{algo}")
ranked_by_query = defaultdict(list)
for r in results:
    ranked_by_query[r.query_id].append((r.doc_id, r.score))

# Sort
for qid in ranked_by_query:
    ranked_by_query[qid].sort(key=lambda x: -x[1])
    ranked_by_query[qid] = [doc_id for doc_id, _ in ranked_by_query[qid]]

map_scores, p_at_k, r_at_k, ndcg_scores = [], [], [], []
for qid, rel_docs in qrels.items():
    ranked = ranked_by_query.get(qid, [])
    if not ranked:
        continue
    map_scores.append(average_precision(ranked, rel_docs))
    p_at_k.append(precision_at_k(ranked, rel_docs, k))
    r_at_k.append(recall_at_k(ranked, rel_docs, k))
    ndcg_scores.append(ndcg_at_k(ranked, rel_docs, k))

metrics = {{
    "MAP": round(np.mean(map_scores), 4),
    "Precision@10": round(np.mean(p_at_k), 4),
    "Recall@10": round(np.mean(r_at_k), 4),
    "NDCG@10": round(np.mean(ndcg_scores), 4),
}}
pd.DataFrame(metrics, index=["{display_name}"]).T
"""))

    cells.append(nbf.v4.new_code_cell(f"""
# Plot for {display_name}
pd.DataFrame(metrics, index=["{display_name}"]).plot(
    kind="bar", figsize=(6,4), legend=False, title="{display_name} Evaluation"
)
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.grid(axis="y")
plt.tight_layout()
plt.show()
"""))

# Write notebook
nb['cells'] = cells
with open(notebook_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"✅ Notebook generated: {notebook_path}")
