from models.search_result import SearchResult

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