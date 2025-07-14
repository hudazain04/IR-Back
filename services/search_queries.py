


from database import SessionLocal
from repositories import query_repo, search_result_repo
from services.search_service import SearchService
import time
from services.bm_25_par import run_bm25_sequential

search_service = SearchService()

BATCH_SIZE = 500 

def process_query(query, config, dataset_name, top_k):
    label = config["label"]
    algo = config["algo"]
    with_index = config["with_index"]
    with_additional = config["with_additional"]

    print(f"▶️ Starting query {query.query_id} on {label.upper()}")

    try:
        start_time = time.time()
        results = search_service.search(
            query=query.raw_text,
            algorithm=algo,
            dataset_name=dataset_name,
            top_k=top_k,
            with_index=with_index,
            with_additional=with_additional,
        )
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"⏱️ Search time for query {query.query_id} on {label.upper()}: {elapsed:.2f} seconds")

        results_to_insert = []
        for rank, result in enumerate(results, start=1):
            results_to_insert.append({
                "query_id": query.query_id,
                "doc_id": result["doc_id"],
                "rank": rank,
                "score": result["score"],
                "algorithm": label,
                "dataset": dataset_name
            })

        return results_to_insert

    except Exception as e:
        print(f"❌ Failed query {query.query_id} on {label.upper()}: {e}")
        return []


def run_search_for_all_algorithms(dataset_name, top_k=10):
    db = SessionLocal()
    start_hole_time = time.time()
    queries = query_repo.get_queries_by_source(db, dataset_name)
    db.close()

    search_variants = [
        # {"algo": "vsm", "with_index": True, "with_additional": False, "label": "vsm_index"},
        # {"algo": "word2vec", "with_index": False, "with_additional": False, "label": "word2vec_plain"},
        # {"algo": "word2vec", "with_index": False, "with_additional": True,  "label": "word2vec_faiss"},
        # {"algo": "hybrid", "with_index": False, "with_additional": False, "label": "hybrid_plain"},
        # {"algo": "hybrid", "with_index": False, "with_additional": True,  "label": "hybrid_faiss"},
        {"algo": "bm25", "with_index": True, "with_additional": False, "label": "bm25"},
    ]

    
    for config in search_variants:
        label = config["label"]
        print(f"\n🔍 Running {label.upper()} search on {len(queries)} queries...")


        if config["algo"] == "bm25":
            print(f"\n🚀 Running BM25 using multiprocessing...")
            result = run_bm25_sequential(dataset_name)
            print(f"✅ BM25 finished: {result}")
            continue 

        print(f"\n🔍 Running {label.upper()} search on {len(queries)} queries...")

        db_clear = SessionLocal()
        search_result_repo.clear_results(db_clear, algorithm=label, dataset=dataset_name)
        db_clear.commit()
        db_clear.close()

        all_results_to_insert = []

        for query in queries:
            results = process_query(query, config, dataset_name, top_k)
            all_results_to_insert.extend(results)

        if all_results_to_insert:
            print(f"📦 Preparing to store {len(all_results_to_insert)} results for {label.upper()} in batches of {BATCH_SIZE}")
            start_insert_time = time.time()

            for i in range(0, len(all_results_to_insert), BATCH_SIZE):
                batch = all_results_to_insert[i:i + BATCH_SIZE]
                db_insert = SessionLocal()
                try:
                    search_result_repo.bulk_upsert(db_insert, batch)
                    db_insert.commit()
                    print(f"✅ Inserted batch {i // BATCH_SIZE + 1} of size {len(batch)}")
                except Exception as e:
                    print(f"❌ Failed to insert batch {i // BATCH_SIZE + 1}: {e}")
                    db_insert.rollback()
                finally:
                    db_insert.close()

            end_insert_time = time.time()
            print(f"✅ Finished storing all results for {label.upper()} in {end_insert_time - start_insert_time:.2f} seconds")
        else:
            print(f"⚠️ No results to store for {label.upper()}")

    end_hole_time = time.time()
    total_time = end_hole_time - start_hole_time
    print(f"🕒 Total execution time for all algorithms: {total_time:.2f} seconds")
    return {"status": "success", "message": "Search completed for all configurations."}