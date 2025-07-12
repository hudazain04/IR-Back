import os
import joblib
import faiss

def build_faiss_for_dataset(dataset_name: str):
    base_path = "vector_store_word2vec"
    faiss_base_path = os.path.join("vector_store_faiss", dataset_name)
    os.makedirs(faiss_base_path, exist_ok=True)

    matrix = joblib.load(os.path.join(base_path, dataset_name, "w2v_matrix.joblib"))
    doc_ids = joblib.load(os.path.join(base_path, dataset_name, "doc_ids.joblib"))
    doc_texts = joblib.load(os.path.join(base_path, dataset_name, "doc_texts.joblib"))

    dense_matrix = matrix.astype("float32")
    faiss.normalize_L2(dense_matrix)
    dim = dense_matrix.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(dense_matrix)

    faiss.write_index(index, os.path.join(faiss_base_path, "faiss.index"))
    joblib.dump(doc_ids, os.path.join(faiss_base_path, "doc_ids.joblib"))
    joblib.dump(doc_texts, os.path.join(faiss_base_path, "doc_texts.joblib"))

    return {"status": "success", "message": f"FAISS index built and saved for dataset '{dataset_name}'"}
