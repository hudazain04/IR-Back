from models.search_result import SearchResult
from sqlalchemy.dialects.mysql import insert


def  upsert_result(db, query_id, doc_id, rank, score, algorithm, dataset):
    result = SearchResult(
        query_id=query_id,
        doc_id=doc_id,
        rank=rank,
        score=score,
        algorithm=algorithm,
        dataset=dataset
    )
    db.add(result)

def commit(db):
    db.commit()



def bulk_upsert(db, results: list[dict]):
    if not results:
        return

    stmt = insert(SearchResult).values(results)

    update_stmt = {  # exclude primary keys here
        "rank": stmt.inserted.rank,
        "score": stmt.inserted.score,
        "algorithm": stmt.inserted.algorithm,
        "dataset": stmt.inserted.dataset,
    }

    upsert_stmt = stmt.on_duplicate_key_update(**update_stmt)

    db.execute(upsert_stmt)
    db.commit()


def clear_results(db, algorithm=None, dataset=None):
    query = db.query(SearchResult)
    if algorithm:
        query = query.filter(SearchResult.algorithm == algorithm)
    if dataset:
        query = query.filter(SearchResult.dataset == dataset)
    query.delete()
    db.commit()

def get_results_by_algorithm(db, algorithm: str, dataset_name: str):
    return db.query(SearchResult).filter(
        SearchResult.algorithm == algorithm,
        SearchResult.dataset == dataset_name
    ).all()