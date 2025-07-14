from multiprocessing import Pool, cpu_count
from database import SessionLocal
from repositories import query_repo, search_result_repo
from services.search_service import SearchService
import time

BATCH_SIZE = 100  # for database insertion
QUERY_CHUNK_SIZE = 200  # queries per worker
WORKERS = 3  # number of parallel processes

def process_bm25_chunk(args):
    """Worker function for processing BM25 queries chunk."""
    queries, dataset_name = args
    search_service = SearchService()  # each process must create its own instance
    label = "bm25"

    results_to_insert = []
    for query in queries:
        try:
            start_time = time.time()
            results = search_service.search(
                query=query.raw_text,
                algorithm="bm25",
                dataset_name=dataset_name,
                top_k=10,
                with_index=False,
                with_additional=False,
            )
            end_time = time.time()
            print(f"⏱️ Query {query.query_id} processed in {end_time - start_time:.2f}s")

            for rank, result in enumerate(results, start=1):
                results_to_insert.append({
                    "query_id": query.query_id,
                    "doc_id": result["doc_id"],
                    "rank": rank,
                    "score": result["score"],
                    "algorithm": label,
                    "dataset": dataset_name
                })
        except Exception as e:
            print(f"❌ Failed processing query {query.query_id}: {e}")

    return results_to_insert


def run_bm25_parallel(dataset_name):
    db = SessionLocal()
    queries = query_repo.get_queries_by_source(db, dataset_name)
    db.close()

    print(f"🔍 Total BM25 queries: {len(queries)}")
    print(f"🧠 Using {WORKERS} workers with chunk size {QUERY_CHUNK_SIZE}")

    # Clear existing results
    db_clear = SessionLocal()
    search_result_repo.clear_results(db_clear, algorithm="bm25", dataset=dataset_name)
    db_clear.commit()
    db_clear.close()

    start_all_time = time.time()

    # Create query chunks
    query_chunks = [queries[i:i + QUERY_CHUNK_SIZE] for i in range(0, len(queries), QUERY_CHUNK_SIZE)]
    args = [(chunk, dataset_name) for chunk in query_chunks]

    # Process with multiprocessing
    all_results = []
    with Pool(WORKERS) as pool:
        for results in pool.imap_unordered(process_bm25_chunk, args):
            all_results.extend(results)

    print(f"✅ BM25 processing done. Total results: {len(all_results)}")

    for i in range(0, len(all_results), BATCH_SIZE):
        batch = all_results[i:i + BATCH_SIZE]
        db_insert = SessionLocal()
        try:
            search_result_repo.bulk_upsert(db_insert, batch)
            db_insert.commit()
            print(f"✅ Inserted result batch {i // BATCH_SIZE + 1}")
        except Exception as e:
            print(f"❌ Failed to insert batch {i // BATCH_SIZE + 1}: {e}")
            db_insert.rollback()
        finally:
            db_insert.close()

    end_all_time = time.time()
    print(f"🕒 Total execution time: {end_all_time - start_all_time:.2f} seconds")
    return {"status": "success", "message": "BM25 search completed with multiprocessing."}
