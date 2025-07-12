from sqlalchemy.orm import Session
from repositories import query_repo, search_result_repo
from services.search_service import SearchService
from database import SessionLocal
from fastapi import APIRouter

router = APIRouter()

# def run_search_for_all_algorithms(dataset_name, top_k=10, with_index=False, with_additional=False):
#     db: Session = SessionLocal()
#     queries = query_repo.get_queries_by_source(db, dataset_name)
#     search_service = SearchService()

#     algorithms = ["vsm", "word2vec", "bm25", "hybrid"]

#     for algo in algorithms:
#         print(f"\n🔍 Running {algo.upper()} search on {len(queries)} queries...")
#         search_result_repo.clear_results(db, algorithm=algo, dataset=dataset_name)

#         for query in queries:
#             try:
#                 results = search_service.search(
#                     query=query.raw_text,
#                     algorithm=algo,
#                     dataset_name=dataset_name,
#                     top_k=top_k,
#                     with_index=with_index,
#                     with_additional=with_additional
#                 )

#                 for rank, result in enumerate(results, start=1):
#                     search_result_repo.upsert_result(
#                         db=db,
#                         query_id=query.query_id,
#                         doc_id=result["doc_id"],
#                         rank=rank,
#                         score=result["score"],
#                         algorithm=algo,
#                         dataset=dataset_name
#                     )
#             except Exception as e:
#                 print(f"❌ Failed query {query.doc_id} on {algo.upper()}: {e}")

#         search_result_repo.commit(db)
#         print(f"✅ Stored results for {algo.upper()}")
#     return {"status": "success", "message":f"✅ Stored results for {algorithms}"}


# # if __name__ == "__main__":
# #     run_search_for_all_algorithms(dataset_name="cranfield", top_k=10)
def run_search_for_all_algorithms(dataset_name, top_k=10):
    db: Session = SessionLocal()
    queries = query_repo.get_queries_by_source(db, dataset_name)
    search_service = SearchService()

    search_variants = [
        {"algo": "vsm", "with_index": True, "with_additional": False, "label": "vsm_index"},
        {"algo": "word2vec", "with_index": False, "with_additional": False, "label": "word2vec_plain"},
        {"algo": "word2vec", "with_index": False, "with_additional": True,  "label": "word2vec_faiss"},
        {"algo": "hybrid", "with_index": False, "with_additional": False, "label": "hybrid_plain"},
        {"algo": "hybrid", "with_index": False, "with_additional": True,  "label": "hybrid_faiss"},
        {"algo": "bm25", "with_index": False, "with_additional": False, "label": "bm25"},  # optional
    ]

    for config in search_variants:
        algo = config["algo"]
        label = config["label"]
        with_index = config["with_index"]
        with_additional = config["with_additional"]

        print(f"\n🔍 Running {label.upper()} search on {len(queries)} queries...")
        search_result_repo.clear_results(db, algorithm=label, dataset=dataset_name)
        # queries = queries[:10]
        for query in queries:
            print(f"🧩 [{label}] Processing query {query.query_id}/{queries.count}")
            try:
                results = search_service.search(
                    query=query.raw_text,
                    algorithm=algo,
                    dataset_name=dataset_name,
                    top_k=top_k,
                    with_index=with_index,
                    with_additional=with_additional
                )

                for rank, result in enumerate(results, start=1):
                    search_result_repo.upsert_result(
                        db=db,
                        query_id=query.query_id,
                        doc_id=result["doc_id"],
                        rank=rank,
                        score=result["score"],
                        algorithm=label,  # <-- important: use distinct name
                        dataset=dataset_name
                    )
            except Exception as e:
                print(f"❌ Failed query {query.query_id} on {label.upper()}: {e}")

        search_result_repo.commit(db)
        print(f"✅ Stored results for {label.upper()}")

    return {"status": "success", "message": "Search completed for all configurations."}
# import concurrent.futures
# from sqlalchemy.orm import Session
# from database import SessionLocal
# from repositories import query_repo, search_result_repo
# from services.search_service import SearchService
# import os
# # max_workers = os.cpu_count() or 6
# max_workers = min(32,( os.cpu_count() or 6)*4)

# def run_search_for_all_algorithms(dataset_name, top_k=10):
#     db: Session = SessionLocal()
#     queries = query_repo.get_queries_by_source(db, dataset_name)
#     search_service = SearchService()

#     search_variants = [
#         {"algo": "vsm", "with_index": True, "with_additional": False, "label": "vsm_index"},
#         {"algo": "word2vec", "with_index": False, "with_additional": False, "label": "word2vec_plain"},
#         {"algo": "word2vec", "with_index": False, "with_additional": True,  "label": "word2vec_faiss"},
#         {"algo": "hybrid", "with_index": False, "with_additional": False, "label": "hybrid_plain"},
#         {"algo": "hybrid", "with_index": False, "with_additional": True,  "label": "hybrid_faiss"},
#         {"algo": "bm25", "with_index": False, "with_additional": False, "label": "bm25"},
#     ]

#     for config in search_variants:
#         algo = config["algo"]
#         label = config["label"]
#         with_index = config["with_index"]
#         with_additional = config["with_additional"]

#         total_queries = len(queries)
#         print(f"\n🔍 Running {label.upper()} search on {total_queries} queries...")
#         search_result_repo.clear_results(db, algorithm=label, dataset=dataset_name)

#         def process_query(index, query):
#             try:
#                 print(f"🧩 [{label}] Processing query {index + 1}/{total_queries} (ID={query.query_id})")
#                 local_db = SessionLocal()
#                 results = search_service.search(
#                     query=query.raw_text,
#                     algorithm=algo,
#                     dataset_name=dataset_name,
#                     top_k=top_k,
#                     with_index=with_index,
#                     with_additional=with_additional
#                 )
#                 for rank, result in enumerate(results, start=1):
#                     search_result_repo.upsert_result(
#                         db=local_db,
#                         query_id=query.query_id,
#                         doc_id=result["doc_id"],
#                         rank=rank,
#                         score=result["score"],
#                         algorithm=label,
#                         dataset=dataset_name
#                     )
#                 local_db.commit()
#                 local_db.close()
#             except Exception as e:
#                 print(f"❌ Failed query {query.query_id} on {label.upper()}: {e}")

#         # Use submit to preserve index info
#         with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
#             futures = [
#                 executor.submit(process_query, i, q)
#                 for i, q in enumerate(queries)
#             ]
#             # Optional: wait for completion
#             concurrent.futures.wait(futures)

#         print(f"✅ Stored results for {label.upper()}")

#     return {"status": "success", "message": "Search completed for all configurations."}