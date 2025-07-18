import joblib
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from services.processor import TextProcessor
from rank_bm25 import BM25Okapi
import faiss
from collections import Counter
from sklearn.preprocessing import normalize
import time
from services.word2vec_representation import Word2VecRepresentation
import heapq

class SearchService:
    def __init__(self):
        self.processor = TextProcessor()
        self.word2vecRep = Word2VecRepresentation()
        self.word2vec_loaded = False
        self.tfidf_cache = {}
        self.word2vec_cache = {}
        self.bm25_cache = {}
        self.faiss_cache = {}
        self.inverted_index_cache = {}
        self.bm25_score_cache = {}

    def load_tfidf_assets(self, dataset_name: str):
        if dataset_name in self.tfidf_cache:
            return self.tfidf_cache[dataset_name]

        base_dir = os.path.join("vector_store", dataset_name)

        try:
            vectorizer = joblib.load(os.path.join(base_dir, "vectorizer.joblib"))
            matrix = joblib.load(os.path.join(base_dir, "tfidf_matrix.joblib"))
            doc_ids = joblib.load(os.path.join(base_dir, "doc_ids.joblib"))
            doc_texts = joblib.load(os.path.join(base_dir, "doc_texts.joblib"))
        except FileNotFoundError as e:
            raise ValueError(f"Missing TF-IDF joblib files for dataset '{dataset_name}'") from e

        self.tfidf_cache[dataset_name] = (vectorizer, matrix, doc_ids, doc_texts)
        return self.tfidf_cache[dataset_name]

    def load_faiss_assets(self, dataset_name: str):
        if dataset_name in self.faiss_cache:
            return self.faiss_cache[dataset_name]

        base_path = os.path.join("vector_store_faiss", dataset_name)

        try:
            index = faiss.read_index(os.path.join(base_path, "faiss.index"))
            doc_ids = joblib.load(os.path.join(base_path, "doc_ids.joblib"))
            doc_texts = joblib.load(os.path.join(base_path, "doc_texts.joblib"))
        except Exception as e:
            raise ValueError(f"FAISS assets missing for dataset '{dataset_name}'") from e

        self.faiss_cache[dataset_name] = (index, doc_ids, doc_texts)
        return self.faiss_cache[dataset_name]


    def load_word2vec_assets(self, dataset_name: str):
        if dataset_name in self.word2vec_cache:
            self.w2v_model, self.w2v_matrix, self.doc_ids, self.doc_texts = self.word2vec_cache[dataset_name]
            return self.word2vec_cache[dataset_name]

        base_dir = os.path.join("vector_store_word2vec", dataset_name)

        try:
            w2v_model: Word2Vec = joblib.load(os.path.join(base_dir, "w2v_model.joblib"))
            w2v_matrix = joblib.load(os.path.join(base_dir, "w2v_matrix.joblib"))
            doc_ids = joblib.load(os.path.join(base_dir, "doc_ids.joblib"))
            doc_texts = joblib.load(os.path.join(base_dir, "doc_texts.joblib"))
        except FileNotFoundError as e:
            raise ValueError("Word2Vec joblib files not found.") from e

        w2v_matrix = normalize(w2v_matrix, axis=1) 

        self.w2v_model = w2v_model
        self.w2v_matrix = w2v_matrix
        self.doc_ids = doc_ids
        self.doc_texts = doc_texts

        self.word2vec_cache[dataset_name] = (w2v_model, w2v_matrix, doc_ids, doc_texts)
        return self.word2vec_cache[dataset_name]


    def load_inverted_index(self, dataset_name: str):
        if dataset_name in self.inverted_index_cache:
            return self.inverted_index_cache[dataset_name]

        path = os.path.join("vector_store_inverted", dataset_name, "inverted_index.joblib")
        if not os.path.exists(path):
            raise ValueError(f"Inverted index not found for dataset '{dataset_name}'")

        inverted_index = joblib.load(path)
        self.inverted_index_cache[dataset_name] = inverted_index
        return inverted_index

    def load_bm25_assets(self, dataset_name: str):
        if dataset_name in self.bm25_cache:
            return self.bm25_cache[dataset_name]

        base_path = os.path.join("vector_store_bm25", dataset_name)

        try:
            bm25 = joblib.load(os.path.join(base_path, "bm25_model.joblib"))
            doc_ids = joblib.load(os.path.join(base_path, "doc_ids.joblib"))
            doc_texts = joblib.load(os.path.join(base_path, "doc_texts.joblib"))
        except FileNotFoundError as e:
            raise ValueError("BM25 joblib files not found.") from e

        self.bm25_cache[dataset_name] = (bm25, doc_ids, doc_texts)
        return self.bm25_cache[dataset_name]


    def search_faiss(self, query: str, dataset_name: str, top_k=5, with_index=False):
        vectorizer, matrix, doc_ids_full, doc_texts_full = self.load_tfidf_assets(dataset_name)
        index, faiss_doc_ids, faiss_doc_texts = self.load_faiss_assets(dataset_name)

        tokens = self.processor.normalize(query)
        query_vector = vectorizer.transform([" ".join(tokens)]).toarray().astype("float32")

        if with_index:
            inverted_index = self.load_inverted_index(dataset_name)
            matched_ids = set()
            for token in tokens:
                matched_ids.update(inverted_index.get(token, []))

            filtered_data = [(i, doc_id, text)
                            for i, (doc_id, text) in enumerate(zip(doc_ids_full, doc_texts_full))
                            if doc_id in matched_ids]

            if not filtered_data:
                return []

            indices, filtered_ids, filtered_texts = zip(*filtered_data)
            filtered_matrix = matrix[list(indices)].toarray().astype("float32")

            # Build a new temporary FAISS index for filtered data
            cache_key = (dataset_name, frozenset(matched_ids))
            if cache_key in self.filtered_faiss_cache:
                faiss_index, doc_ids, doc_texts = self.filtered_faiss_cache[cache_key]
            else:
                faiss_index = faiss.IndexFlatL2(filtered_matrix.shape[1])
                faiss_index.add(filtered_matrix)
                doc_ids, doc_texts = list(filtered_ids), list(filtered_texts)
                self.filtered_faiss_cache[cache_key] = (faiss_index, doc_ids, doc_texts)

            distances, neighbors = faiss_index.search(query_vector, top_k)

            doc_ids, doc_texts = list(filtered_ids), list(filtered_texts)
        else:
            distances, neighbors = index.search(query_vector, top_k)
            doc_ids, doc_texts = faiss_doc_ids, faiss_doc_texts

        return [
            {
                "doc_id": doc_ids[neighbors[0][i]],
                "text": doc_texts[neighbors[0][i]],
                "distance": float(distances[0][i]), 
                "score": float(1 / (1 + distances[0][i])) 
            }
            for i in range(len(neighbors[0]))
        ]

    

    




    def search_bm25(self, query: str, dataset_name: str, top_k=5, with_index=True):
        start_total = time.perf_counter()
        t0 = time.perf_counter()
        if dataset_name not in self.bm25_cache:
            self.bm25_cache[dataset_name] = self.load_bm25_assets(dataset_name)
        bm25, doc_ids, doc_texts = self.bm25_cache[dataset_name]
        print(f"⏱️ Load BM25 assets: {time.perf_counter() - t0:.4f}s")

        t0 = time.perf_counter()
        tokens = self.processor.normalize(query)
        print(f"⏱️ Tokenization: {time.perf_counter() - t0:.4f}s")
        if not tokens:
            return []

        t0 = time.perf_counter()
        id_to_index = {doc_id: i for i, doc_id in enumerate(doc_ids)}
        print(f"⏱️ ID to index map: {time.perf_counter() - t0:.4f}s")

        if with_index:
            t0 = time.perf_counter()
            if dataset_name not in self.inverted_index_cache:
                self.inverted_index_cache[dataset_name] = self.load_inverted_index(dataset_name)
            inverted_index = self.inverted_index_cache[dataset_name]
            print(f"⏱️ Load inverted index: {time.perf_counter() - t0:.4f}s")

            t0 = time.perf_counter()
            matched_ids = set()
            for token in tokens:
                matched_ids.update(inverted_index.get(token, []))
            print(f"⏱️ Matching doc IDs from index: {time.perf_counter() - t0:.4f}s")

            if not matched_ids:
                return []

            t0 = time.perf_counter()
            matched_indices = [id_to_index[doc_id] for doc_id in matched_ids if doc_id in id_to_index]
            print(f"⏱️ Map matched IDs to indices: {time.perf_counter() - t0:.4f}s")

            if not matched_indices:
                return []

            t0 = time.perf_counter()
            cache_key = (dataset_name, tuple(sorted(tokens)), frozenset(matched_ids))
            scores = self.bm25_score_cache.get(cache_key)
            if scores is None:
                scores = bm25.get_batch_scores(tokens, matched_indices)
                self.bm25_score_cache[cache_key] = scores
            print(f"⏱️ Compute BM25 scores (with_index): {time.perf_counter() - t0:.4f}s")

            filtered_ids = [doc_ids[i] for i in matched_indices]
            filtered_texts = [doc_texts[i] for i in matched_indices]
        else:
            t0 = time.perf_counter()
            scores = bm25.get_scores(tokens)
            print(f"⏱️ Compute BM25 scores (full corpus): {time.perf_counter() - t0:.4f}s")

            if scores is None or (hasattr(scores, 'size') and scores.size == 0) or (isinstance(scores, list) and len(scores) == 0):
                return []

            filtered_ids = doc_ids
            filtered_texts = doc_texts

        if scores is None or len(scores) == 0:
            return []

        t0 = time.perf_counter()
        num_scores = len(scores)
        top_indices = heapq.nlargest(top_k, range(num_scores), key=lambda i: scores[i])
        print(f"⏱️ Top-k selection: {time.perf_counter() - t0:.4f}s")

        t0 = time.perf_counter()
        results = [
            {
                "doc_id": filtered_ids[i],
                "text": filtered_texts[i],
                "score": float(scores[i])
            }
            for i in top_indices
        ]
        print(f"⏱️ Format results: {time.perf_counter() - t0:.4f}s")

        print(f"✅ Total search time: {time.perf_counter() - start_total:.4f}s\n")
        return results




    

    def filter_documents_by_inverted_index(self, query: str, dataset_name: str, doc_ids, doc_texts, matrix, max_docs=300):
        inverted_index = self.load_inverted_index(dataset_name)
        tokens = set(self.processor.normalize(query))

        doc_score = Counter()
        for token in tokens:
            for doc_id in inverted_index.get(token, []):
                doc_score[doc_id] += 1

        top_doc_ids = {doc_id for doc_id, _ in doc_score.most_common(max_docs)}
        filtered_data = [
            (i, doc_id, text)
            for i, (doc_id, text) in enumerate(zip(doc_ids, doc_texts))
            if doc_id in top_doc_ids
        ]

        if not filtered_data:
            return [], [], [], None

        indices, filtered_ids, filtered_texts = zip(*filtered_data)
        filtered_matrix = matrix[list(indices)]

        return list(filtered_ids), list(filtered_texts), list(indices), filtered_matrix



    def search_vsm(self, query: str, dataset_name: str, top_k=5, with_index=False):
        vectorizer, tfidf_matrix, doc_ids, doc_texts = self.load_tfidf_assets(dataset_name)

        if with_index:
            doc_ids, doc_texts, indices, tfidf_matrix = self.filter_documents_by_inverted_index(
                query, dataset_name, doc_ids, doc_texts, tfidf_matrix)
            if not doc_ids:
                return []

        tokens = self.processor.normalize(query)
        query_vector = vectorizer.transform([" ".join(tokens)])
        similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
        top_indices = similarities.argsort()[::-1][:top_k]

        return [
            {"doc_id": doc_ids[i], "text": doc_texts[i], "score": float(similarities[i])}
            for i in top_indices
        ]

    


    def search_word2vec(self, query: str, dataset_name: str, top_k=5, with_additional=False):
        self.load_word2vec_assets(dataset_name)


        tokens = self.processor.normalize(query)


        vectors = [self.w2v_model.wv[word] for word in tokens if word in self.w2v_model.wv]

        if vectors:
            query_vector = np.mean(vectors, axis=0)
        else:
            query_vector = np.zeros(self.w2v_model.vector_size)

        query_vector = query_vector.reshape(1, -1)

        if with_additional:
            index, doc_ids, doc_texts = self.load_faiss_assets(dataset_name)

            faiss.normalize_L2(query_vector)
            
            D, I = index.search(query_vector, top_k)

            return [
                {"doc_id": doc_ids[i], "text": doc_texts[i], "score": float(D[0][rank])}
                for rank, i in enumerate(I[0])
            ]


        similarities = np.dot(self.w2v_matrix , query_vector.T).flatten()

        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [
                {"doc_id": self.doc_ids[i], "text": self.doc_texts[i], "score": float(similarities[i])}
                for i in top_indices
            ]


    

    def search_hybrid(self, query: str, dataset_name: str, top_k=5, alpha=0.5, beta=0.5, with_index=False, with_additional=False):
     
        vectorizer, tfidf_matrix, doc_ids, doc_texts = self.load_tfidf_assets(dataset_name)
        self.load_word2vec_assets(dataset_name)

        if with_index:
            doc_ids, doc_texts, indices, tfidf_matrix = self.filter_documents_by_inverted_index(
                query, dataset_name, doc_ids, doc_texts, tfidf_matrix
            )
            if not doc_ids:
                return []
            w2v_matrix = self.w2v_matrix[indices]
        else:
            w2v_matrix = self.w2v_matrix

        tokens = self.processor.normalize(query)

        tfidf_vector = vectorizer.transform([" ".join(tokens)])
        sim_tfidf = tfidf_matrix.dot(tfidf_vector.T).toarray().ravel()
        sim_tfidf /= sim_tfidf.max() or 1

        w2v_query_vec = np.mean(
            [self.w2v_model.wv[word] for word in tokens if word in self.w2v_model.wv]
            or [np.zeros(self.w2v_model.vector_size)],
            axis=0
        ).astype("float32")

        if with_additional:
            faiss.normalize_L2(w2v_query_vec.reshape(1, -1))
            index, faiss_doc_ids, faiss_doc_texts = self.load_faiss_assets(dataset_name)
            distances, neighbors = index.search(w2v_query_vec.reshape(1, -1), top_k * 10) 

            id_to_score = {doc_id: score for doc_id, score in zip(faiss_doc_ids, distances[0])}
            sim_w2v = np.array([id_to_score.get(doc_id, 0.0) for doc_id in doc_ids])
        else:
            sim_w2v = np.dot(w2v_matrix, w2v_query_vec)

        sim_w2v /= sim_w2v.max() or 1

        final_scores = alpha * sim_tfidf + beta * sim_w2v
        top_indices = final_scores.argsort()[::-1][:top_k]

        return [
            {"doc_id": doc_ids[i], "text": doc_texts[i], "score": float(final_scores[i])}
            for i in top_indices
        ]


    def search(self, query: str, algorithm: str = "vsm", dataset_name: str = "", top_k=5, with_index=False, with_additional=False):
        algorithm = algorithm.lower()
        if algorithm == "vsm":
            return self.search_vsm(query, dataset_name, top_k, with_index)
        elif algorithm == "word2vec":
            return self.search_word2vec(query, dataset_name, top_k, with_additional)
        elif algorithm == "hybrid":
            return self.search_hybrid(query, dataset_name, top_k, alpha=0.5, with_index=with_index,  with_additional=with_additional)
        elif algorithm == "bm25":
            return self.search_bm25(query, dataset_name, top_k, with_index)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
