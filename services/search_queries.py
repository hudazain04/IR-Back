# from sqlalchemy.orm import Session
# from repositories import query_repo, search_result_repo
# from services.search_service import SearchService
# from database import SessionLocal
# from fastapi import APIRouter

# router = APIRouter()

# def run_search_for_all_algorithms(dataset_name, db , top_k=10 ):
#     queries = query_repo.get_queries_by_source(db, dataset_name)
#     search_service = SearchService()

#     search_variants = [
#         {"algo": "vsm", "with_index": True, "with_additional": False, "label": "vsm_index"},
#         {"algo": "word2vec", "with_index": False, "with_additional": False, "label": "word2vec_plain"},
#         {"algo": "word2vec", "with_index": False, "with_additional": True,  "label": "word2vec_faiss"},
#         {"algo": "hybrid", "with_index": False, "with_additional": False, "label": "hybrid_plain"},
#         {"algo": "hybrid", "with_index": False, "with_additional": True,  "label": "hybrid_faiss"},
#         {"algo": "bm25", "with_index": False, "with_additional": False, "label": "bm25"},  # optional
#     ]

#     for config in search_variants:
#         algo = config["algo"]
#         label = config["label"]
#         with_index = config["with_index"]
#         with_additional = config["with_additional"]

#         print(f"\n🔍 Running {label.upper()} search on {len(queries)} queries...")
#         search_result_repo.clear_results(db, algorithm=label, dataset=dataset_name)
#         # queries = queries[:10]
#         row_count = 0
#         for query in queries:
#             print(f"🧩 [{label}] Processing query {query.query_id}/{queries.count}")
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
#                         algorithm=label,  # <-- important: use distinct name
#                         dataset=dataset_name
#                     )
#                     row_count += 1
#                     if row_count % 1000 == 0:
#                         search_result_repo.commit(db)
#             except Exception as e:
#                 print(f"❌ Failed query {query.query_id} on {label.upper()}: {e}")

#         search_result_repo.commit(db)
#         print(f"✅ Stored results for {label.upper()}")

#     return {"status": "success", "message": "Search completed for all configurations."}


# from concurrent.futures import ThreadPoolExecutor, as_completed
# from database import SessionLocal
# from repositories import query_repo, search_result_repo
# from services.search_service import SearchService
# import os

# def process_query(query, config, dataset_name, top_k):
#     db = SessionLocal()  # create new session for each thread
#     search_service = SearchService()
#     label = config["label"]
#     algo = config["algo"]
#     with_index = config["with_index"]
#     with_additional = config["with_additional"]

#     print(f"▶️ Starting query {query.query_id} on {label.upper()}")

#     row_count = 0
#     try:
#         results = search_service.search(
#             query=query.raw_text,
#             algorithm=algo,
#             dataset_name=dataset_name,
#             top_k=top_k,
#             with_index=with_index,
#             with_additional=with_additional,
#         )

#         for rank, result in enumerate(results, start=1):
#             search_result_repo.upsert_result(
#                 db=db,
#                 query_id=query.query_id,
#                 doc_id=result["doc_id"],
#                 rank=rank,
#                 score=result["score"],
#                 algorithm=label,
#                 dataset=dataset_name,
#             )
#             row_count += 1
#             if row_count % 1000 == 0:
#                 print(f"  💾 Done 1000 results for query {query.query_id}")

#         # Commit leftovers for this query
#         search_result_repo.commit(db)
#         print(f"✅ Finished query {query.query_id} on {label.upper()} with {row_count} results saved.")
#     except Exception as e:
#         print(f"❌ Failed query {query.query_id} on {label.upper()}: {e}")
#     finally:
#         db.close()


# def run_search_for_all_algorithms(dataset_name, top_k=10):
#     queries = query_repo.get_queries_by_source(db, dataset_name)
#     search_variants = [
#         {"algo": "vsm", "with_index": True, "with_additional": False, "label": "vsm_index"},
#         {"algo": "word2vec", "with_index": False, "with_additional": False, "label": "word2vec_plain"},
#         {"algo": "word2vec", "with_index": False, "with_additional": True,  "label": "word2vec_faiss"},
#         {"algo": "hybrid", "with_index": False, "with_additional": False, "label": "hybrid_plain"},
#         {"algo": "hybrid", "with_index": False, "with_additional": True,  "label": "hybrid_faiss"},
#         {"algo": "bm25", "with_index": False, "with_additional": False, "label": "bm25"},
#     ]

#     max_workers = 6

#     for config in search_variants:
#         label = config["label"]
#         print(f"\n🔍 Running {label.upper()} search on {len(queries)} queries...")

#         # Clear previous results for this algorithm and dataset
#         search_result_repo.clear_results(db, algorithm=label, dataset=dataset_name)

#         with ThreadPoolExecutor(max_workers=max_workers) as executor:
#             futures = [
#                 executor.submit(process_query, query, config, dataset_name, top_k)
#                 for query in queries
#             ]
#             for future in as_completed(futures):
#                 # raises exceptions if any occurred in worker
#                 future.result()

#         print(f"✅ Stored results for {label.upper()}")

#     return {"status": "success", "message": "Search completed for all configurations."}





# from concurrent.futures import ThreadPoolExecutor, as_completed
# from database import SessionLocal
# from repositories import query_repo, search_result_repo
# from services.search_service import SearchService

# def process_query(query, config, dataset_name, top_k):
#     db = SessionLocal()
#     search_service = SearchService()

#     label = config["label"]
#     algo = config["algo"]
#     with_index = config["with_index"]
#     with_additional = config["with_additional"]

#     print(f"▶️ Starting query {query.query_id} on {label.upper()}")

#     row_count = 0
#     try:
#         results = search_service.search(
#             query=query.raw_text,
#             algorithm=algo,
#             dataset_name=dataset_name,
#             top_k=top_k,
#             with_index=with_index,
#             with_additional=with_additional,
#         )

#         results_to_insert = []
#         for rank, result in enumerate(results, start=1):
#             results_to_insert.append({
#                 "query_id": query.query_id,
#                 "doc_id": result["doc_id"],
#                 "rank": rank,
#                 "score": result["score"],
#                 "algorithm": label,
#                 "dataset": dataset_name
#             })

#         search_result_repo.bulk_upsert(db, results_to_insert)
#         db.commit()
#         print(f"✅ Finished query {query.query_id} on {label.upper()} with {row_count} results saved.")
#     except Exception as e:
#         print(f"❌ Failed query {query.query_id} on {label.upper()}: {e}")
#     finally:
#         db.close()

# def run_search_for_all_algorithms(dataset_name, top_k=10):
#     db = SessionLocal()
#     queries = query_repo.get_queries_by_source(db, dataset_name)
#     db.close()

#     search_variants = [
#         {"algo": "vsm", "with_index": True, "with_additional": False, "label": "vsm_index"},
#         {"algo": "word2vec", "with_index": False, "with_additional": False, "label": "word2vec_plain"},
#         {"algo": "word2vec", "with_index": False, "with_additional": True,  "label": "word2vec_faiss"},
#         {"algo": "hybrid", "with_index": False, "with_additional": False, "label": "hybrid_plain"},
#         {"algo": "hybrid", "with_index": False, "with_additional": True,  "label": "hybrid_faiss"},
#         {"algo": "bm25", "with_index": False, "with_additional": False, "label": "bm25"},
#     ]

#     max_workers = 3

#     for config in search_variants:
#         label = config["label"]
#         print(f"\n🔍 Running {label.upper()} search on {len(queries)} queries...")

#         db_clear = SessionLocal()
#         search_result_repo.clear_results(db_clear, algorithm=label, dataset=dataset_name)
#         db_clear.close()

#         with ThreadPoolExecutor(max_workers=max_workers) as executor:
#             futures = [
#                 executor.submit(process_query, query, config, dataset_name, top_k)
#                 for query in queries
#             ]
#             for future in as_completed(futures):
#                 future.result()

#         print(f"✅ Stored results for {label.upper()}")

#     return {"status": "success", "message": "Search completed for all configurations."}




# from database import SessionLocal
# from repositories import query_repo, search_result_repo
# from services.search_service import SearchService
# import time

# search_service = SearchService()


# def process_query(query, config, dataset_name, top_k):
#     label = config["label"]
#     algo = config["algo"]
#     with_index = config["with_index"]
#     with_additional = config["with_additional"]

#     print(f"▶️ Starting query {query.query_id} on {label.upper()}")

#     try:
#         start_time = time.time()
#         results = search_service.search(
#             query=query.raw_text,
#             algorithm=algo,
#             dataset_name=dataset_name,
#             top_k=top_k,
#             with_index=with_index,
#             with_additional=with_additional,
#         )
#         end_time = time.time()
#         elapsed = end_time - start_time
#         print(f"⏱️ Search time for query {query.query_id} on {label.upper()}: {elapsed:.2f} seconds")

#         results_to_insert = []
#         for rank, result in enumerate(results, start=1):
#             results_to_insert.append({
#                 "query_id": query.query_id,
#                 "doc_id": result["doc_id"],
#                 "rank": rank,
#                 "score": result["score"],
#                 "algorithm": label,
#                 "dataset": dataset_name
#             })

#         return results_to_insert

#     except Exception as e:
#         print(f"❌ Failed query {query.query_id} on {label.upper()}: {e}")
#         return []  # return empty list on failure


# def run_search_for_all_algorithms(dataset_name, top_k=10):
#     db = SessionLocal()
#     queries = query_repo.get_queries_by_source(db, dataset_name)
#     db.close()  # Close this session because only used for reading queries

#     search_variants = [
#         {"algo": "word2vec", "with_index": False, "with_additional": False, "label": "word2vec_plain"},
#         # {"algo": "hybrid", "with_index": False, "with_additional": False, "label": "hybrid_plain"},
#         # {"algo": "hybrid", "with_index": False, "with_additional": True,  "label": "hybrid_faiss"},
#         # {"algo": "bm25", "with_index": False, "with_additional": False, "label": "bm25"},
#     ]

#     for config in search_variants:
#         label = config["label"]
#         print(f"\n🔍 Running {label.upper()} search on {len(queries)} queries...")

#         # Clear old results for this algorithm and dataset
#         db_clear = SessionLocal()
#         search_result_repo.clear_results(db_clear, algorithm=label, dataset=dataset_name)
#         db_clear.commit()
#         db_clear.close()

#         all_results_to_insert = []

#         for query in queries:
#             results = process_query(query, config, dataset_name, top_k)
#             all_results_to_insert.extend(results)

#         if all_results_to_insert:
#             db_insert = SessionLocal()
#             try:
#                 search_result_repo.bulk_upsert(db_insert, all_results_to_insert)
#                 db_insert.commit()
#                 print(f"✅ Stored {len(all_results_to_insert)} results for {label.upper()}")
#             except Exception as e:
#                 print(f"❌ Failed to store results for {label.upper()}: {e}")
#                 db_insert.rollback()
#             finally:
#                 db_insert.close()
#         else:
#             print(f"⚠️ No results to store for {label.upper()}")

#     return {"status": "success", "message": "Search completed for all configurations."}


from database import SessionLocal
from repositories import query_repo, search_result_repo
from services.search_service import SearchService
import time

search_service = SearchService()

BATCH_SIZE = 500  # adjust based on DB performance

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
    queries = query_repo.get_queries_by_source(db, dataset_name)
    db.close()

    search_variants = [
        {"algo": "word2vec", "with_index": False, "with_additional": False, "label": "word2vec_plain"},
        # Add other algorithms here
    ]

    for config in search_variants:
        label = config["label"]
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

    return {"status": "success", "message": "Search completed for all configurations."}




# def process_query(db, query, config, dataset_name, top_k):
#     # no longer create db here
#     label = config["label"]
#     algo = config["algo"]
#     with_index = config["with_index"]
#     with_additional = config["with_additional"]

#     print(f"▶️ Starting query {query.query_id} on {label.upper()}")

#     try:
#         start_time = time.time()
#         results = search_service.search(
#             query=query.raw_text,
#             algorithm=algo,
#             dataset_name=dataset_name,
#             top_k=top_k,
#             with_index=with_index,
#             with_additional=with_additional,
#         )
#         end_time = time.time()
#         elapsed = end_time - start_time
#         print(f"⏱️ Search time for query {query.query_id} on {label.upper()}: {elapsed:.2f} seconds")

#         results_to_insert = []
#         for rank, result in enumerate(results, start=1):
#             results_to_insert.append({
#                 "query_id": query.query_id,
#                 "doc_id": result["doc_id"],
#                 "rank": rank,
#                 "score": result["score"],
#                 "algorithm": label,
#                 "dataset": dataset_name
#             })

#         search_result_repo.bulk_upsert(db, results_to_insert)
#         db.commit()
#         print(f"✅ Finished query {query.query_id} on {label.upper()} with {len(results_to_insert)} results saved.")
#     except Exception as e:
#         print(f"❌ Failed query {query.query_id} on {label.upper()}: {e}")
#         db.rollback()  # rollback this query's transaction to keep session clean


# def run_search_for_all_algorithms(dataset_name, top_k=10):
#     db = SessionLocal()
#     queries = query_repo.get_queries_by_source(db, dataset_name)
#     db.close()

#     search_variants = [
#         # {"algo": "vsm", "with_index": True, "with_additional": False, "label": "vsm_index"},
#         {"algo": "word2vec", "with_index": False, "with_additional": False, "label": "word2vec_plain"},
#         # {"algo": "word2vec", "with_index": False, "with_additional": True,  "label": "word2vec_faiss"},
#         {"algo": "hybrid", "with_index": False, "with_additional": False, "label": "hybrid_plain"},
#         {"algo": "hybrid", "with_index": False, "with_additional": True,  "label": "hybrid_faiss"},
#         {"algo": "bm25", "with_index": False, "with_additional": False, "label": "bm25"},
#     ]

#     for config in search_variants:
#         label = config["label"]
#         print(f"\n🔍 Running {label.upper()} search on {len(queries)} queries...")

#         db_clear = SessionLocal()
#         search_result_repo.clear_results(db_clear, algorithm=label, dataset=dataset_name)
#         db_clear.close()

#         for query in queries:
#             process_query(query, config, dataset_name, top_k)

#         print(f"✅ Stored results for {label.upper()}")

#     return {"status": "success", "message": "Search completed for all configurations."}

# from multiprocessing import Pool, cpu_count
# from database import SessionLocal
# from repositories import query_repo, search_result_repo
# from services.search_service import SearchService
# import time

# def process_query_mp(args):
#     query, config, dataset_name, top_k = args
#     search_service = SearchService()  # Must be recreated per process
#     try:
#         results = search_service.search(
#             query=query.raw_text,
#             algorithm=config["algo"],
#             dataset_name=dataset_name,
#             top_k=top_k,
#             with_index=config["with_index"],
#             with_additional=config["with_additional"],
#         )
#         results_to_insert = [{
#             "query_id": query.query_id,
#             "doc_id": result["doc_id"],
#             "rank": rank,
#             "score": result["score"],
#             "algorithm": config["label"],
#             "dataset": dataset_name
#         } for rank, result in enumerate(results, start=1)]

#         return results_to_insert
#     except Exception as e:
#         print(f"❌ Error on query {query.query_id}: {e}")
#         return []  # Safe fallback




# def run_search_for_all_algorithms(dataset_name, top_k=10):
#     db = SessionLocal()
#     queries = query_repo.get_queries_by_source(db, dataset_name)
#     db.close()

#     # {"algo": "vsm", "with_index": True, "with_additional": False, "label": "vsm_index"},
#     search_variants = [
#         {"algo": "word2vec", "with_index": False, "with_additional": False, "label": "word2vec_plain"},
#         {"algo": "word2vec", "with_index": False, "with_additional": True,  "label": "word2vec_faiss"},
#         {"algo": "hybrid", "with_index": False, "with_additional": False, "label": "hybrid_plain"},
#         {"algo": "hybrid", "with_index": False, "with_additional": True,  "label": "hybrid_faiss"},
#         {"algo": "bm25", "with_index": False, "with_additional": False, "label": "bm25"},
#     ]

#     for config in search_variants:
#         label = config["label"]
#         print(f"\n🔍 Running {label.upper()} search on {len(queries)} queries...")

#         # Clear previous results for this algorithm
#         db_clear = SessionLocal()
#         search_result_repo.clear_results(db_clear, algorithm=label, dataset=dataset_name)
#         db_clear.close()

#         args_list = [(query, config, dataset_name, top_k) for query in queries]

#         with Pool(processes=3) as pool:
#             all_results_nested = pool.map(process_query_mp, args_list)

#         # Flatten results
#         all_results = [item for sublist in all_results_nested for item in sublist]

#         # Save all to database in bulk
#         db_bulk = SessionLocal()
#         search_result_repo.bulk_upsert(db_bulk, all_results)
#         db_bulk.commit()
#         db_bulk.close()

#         print(f"✅ Stored results for {label.upper()}")

#     return {"status": "success", "message": "Search completed for all configurations."}





# from database import SessionLocal
# from repositories import query_repo, search_result_repo
# from services.search_service import SearchService

# def process_query(query, config, dataset_name, top_k):
#     db = SessionLocal()  # create new session for each query
#     search_service = SearchService()
#     label = config["label"]
#     algo = config["algo"]
#     with_index = config["with_index"]
#     with_additional = config["with_additional"]

#     print(f"▶️ Starting query {query.query_id} on {label.upper()}")

#     row_count = 0
#     try:
#         results = search_service.search(
#             query=query.raw_text,
#             algorithm=algo,
#             dataset_name=dataset_name,
#             top_k=top_k,
#             with_index=with_index,
#             with_additional=with_additional,
#         )

#         for rank, result in enumerate(results, start=1):
#             search_result_repo.upsert_result(
#                 db=db,
#                 query_id=query.query_id,
#                 doc_id=result["doc_id"],
#                 rank=rank,
#                 score=result["score"],
#                 algorithm=label,
#                 dataset=dataset_name,
#             )
#             row_count += 1
#             if row_count % 1000 == 0:
#                 search_result_repo.commit(db)
#                 print(f"  💾 Committed 1000 results for query {query.query_id}")

#         # Commit leftovers for this query
#         search_result_repo.commit(db)
#         print(f"✅ Finished query {query.query_id} on {label.upper()} with {row_count} results saved.")
#     except Exception as e:
#         print(f"❌ Failed query {query.query_id} on {label.upper()}: {e}")
#     finally:
#         db.close()


# def run_search_for_all_algorithms(dataset_name, db, top_k=10):
#     queries = query_repo.get_queries_by_source(db, dataset_name)
#     search_variants = [
#         {"algo": "vsm", "with_index": True, "with_additional": False, "label": "vsm_index"},
#         {"algo": "word2vec", "with_index": False, "with_additional": False, "label": "word2vec_plain"},
#         {"algo": "word2vec", "with_index": False, "with_additional": True,  "label": "word2vec_faiss"},
#         {"algo": "hybrid", "with_index": False, "with_additional": False, "label": "hybrid_plain"},
#         {"algo": "hybrid", "with_index": False, "with_additional": True,  "label": "hybrid_faiss"},
#         {"algo": "bm25", "with_index": False, "with_additional": False, "label": "bm25"},
#     ]

#     for config in search_variants:
#         label = config["label"]
#         print(f"\n🔍 Running {label.upper()} search on {len(queries)} queries...")

#         # Clear previous results for this algorithm and dataset
#         search_result_repo.clear_results(db, algorithm=label, dataset=dataset_name)

#         # Sequential processing (no parallelism)
#         for query in queries:
#             process_query(query, config, dataset_name, top_k)

#         print(f"✅ Stored results for {label.upper()}")

#     return {"status": "success", "message": "Search completed for all configurations."}
