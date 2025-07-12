import json
import requests
import os

# === Step 1: Load Queries ===
queries = {}
with open("qrels/cranfield-queries.txt", encoding="utf-8") as f:
    for line in f:
        qid, query = line.strip().split("\t")
        queries[qid] = query

# === Step 2: Setup ===
algorithms = ["tfidf", "word2vec", "bm25", "hybrid"]
dataset_name = "cranfield"
top_k = 10

# Make sure the results directory exists
os.makedirs("results", exist_ok=True)

# === Step 3: Run Queries ===
for algo in algorithms:
    results = []

    for qid, query in queries.items():
        print(f"🔍 {algo.upper()} | Query {qid}: {query}")
        response = requests.post(
            "http://localhost:8000/search",  # Make sure FastAPI is running
            params={
                "query": query,
                "algorithm": algo,
                "dataset_name": dataset_name,
                "top_k": top_k,
                "with_index": True,
                "with_additional": True
            }
        )

        if response.status_code == 200:
            hits = response.json().get("results", [])
            for hit in hits:
                results.append({
                    "query_id": qid,
                    "doc_id": hit["doc_id"]
                })
        else:
            print(f"❌ Error for query {qid}: {response.status_code}")

    # Save results to file
    output_file = f"results/cranfield/{algo}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"✅ Saved results to {output_file}")
