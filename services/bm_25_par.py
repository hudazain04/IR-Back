from database import SessionLocal
from repositories import query_repo, search_result_repo
from services.search_service import SearchService
import time

BATCH_SIZE = 100 

def run_bm25_sequential(dataset_name):
    db = SessionLocal()
    queries = query_repo.get_queries_by_source(db, dataset_name)
    db.close()

    print(f"🔍 Total BM25 queries: {len(queries)}")

    db_clear = SessionLocal()
    search_result_repo.clear_results(db_clear, algorithm="bm25", dataset=dataset_name)
    db_clear.commit()
    db_clear.close()

    start_all_time = time.time()

    search_service = SearchService()
    label = "bm25"
    all_results = []

    for idx, query in enumerate(queries, start=1):
        try:
            start_time = time.time()
            results = search_service.search(
                query=query.raw_text,
                algorithm="bm25",
                dataset_name=dataset_name,
                top_k=10,
                with_index=True,
                with_additional=False,
            )
            end_time = time.time()
            print(f"⏱️ Query {query.query_id} processed in {end_time - start_time:.2f}s")

            for rank, result in enumerate(results, start=1):
                all_results.append({
                    "query_id": query.query_id,
                    "doc_id": result["doc_id"],
                    "rank": rank,
                    "score": result["score"],
                    "algorithm": label,
                    "dataset": dataset_name
                })
        except Exception as e:
            print(f"❌ Failed processing query {query.query_id}: {e}")

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
    return {"status": "success", "message": "BM25 search completed sequentially."}
