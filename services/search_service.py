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
from services.word2vec_representation import Word2VecRepresentation

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

    # def load_word2vec_assets(self, dataset_name: str):
    #     if dataset_name in self.word2vec_cache:
    #         self.w2v_model, self.w2v_matrix, self.doc_ids, self.doc_texts = self.word2vec_cache[dataset_name]
    #         return self.word2vec_cache[dataset_name]

    #     base_dir = os.path.join("vector_store_word2vec", dataset_name)

    #     try:
    #         w2v_model: Word2Vec = joblib.load(os.path.join(base_dir, "w2v_model.joblib"))
    #         w2v_matrix = joblib.load(os.path.join(base_dir, "w2v_matrix.joblib"))
    #         doc_ids = joblib.load(os.path.join(base_dir, "doc_ids.joblib"))
    #         doc_texts = joblib.load(os.path.join(base_dir, "doc_texts.joblib"))
    #     except FileNotFoundError as e:
    #         raise ValueError("Word2Vec joblib files not found.") from e

    #     self.w2v_model = w2v_model
    #     self.w2v_matrix = w2v_matrix
    #     self.doc_ids = doc_ids
    #     self.doc_texts = doc_texts

    #     self.word2vec_cache[dataset_name] = (w2v_model, w2v_matrix, doc_ids, doc_texts)
    #     return self.word2vec_cache[dataset_name]


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

        w2v_matrix = normalize(w2v_matrix, axis=1)  # Normalize once

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

            # Use the filtered IDs and texts
            doc_ids, doc_texts = list(filtered_ids), list(filtered_texts)
        else:
            distances, neighbors = index.search(query_vector, top_k)
            doc_ids, doc_texts = faiss_doc_ids, faiss_doc_texts

        return [
            {
                "doc_id": doc_ids[neighbors[0][i]],
                "text": doc_texts[neighbors[0][i]],
                "distance": float(distances[0][i]),  # Distance not similarity
                "score": float(1 / (1 + distances[0][i]))  # Optional similarity-like score
            }
            for i in range(len(neighbors[0]))
        ]
    # def search_bm25(self, query: str, dataset_name: str, top_k=5, with_index=False):
    #     bm25, doc_ids, doc_texts = self.load_bm25_assets(dataset_name)
    #     tokens = self.processor.normalize(query)

    #     if with_index:
    #         # Optional: use inverted index to filter docs
    #         inverted_index = self.load_inverted_index(dataset_name)
    #         matched_ids = set()
    #         for token in tokens:
    #             matched_ids.update(inverted_index.get(token, []))

    #         filtered_data = [(i, doc_id, text)
    #                          for i, (doc_id, text) in enumerate(zip(doc_ids, doc_texts))
    #                          if doc_id in matched_ids]

    #         if not filtered_data:
    #             return []

    #         indices, filtered_ids, filtered_texts = zip(*filtered_data)
    #         scores = bm25.get_batch_scores(tokens, list(indices))
    #         doc_ids, doc_texts = list(filtered_ids), list(filtered_texts)
    #     else:
    #         scores = bm25.get_scores(tokens)

    #     top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    #     return [
    #         {"doc_id": doc_ids[i], "text": doc_texts[i], "score": float(scores[i])}
    #         for i in top_indices
    #     ]


    # def search_bm25(self, query: str, dataset_name: str, top_k=5, with_index=False):
    #     bm25, doc_ids, doc_texts = self.load_bm25_assets(dataset_name)
    #     tokens = self.processor.normalize(query)

    #     if with_index:
    #         inverted_index = self.load_inverted_index(dataset_name)
    #         matched_ids = set()
    #         for token in tokens:
    #             matched_ids.update(inverted_index.get(token, []))

    #         if not matched_ids:
    #             return []

    #         id_to_index = {doc_id: i for i, doc_id in enumerate(doc_ids)}
    #         filtered_indices = [id_to_index[doc_id] for doc_id in matched_ids if doc_id in id_to_index]

    #         if not filtered_indices:
    #             return []

    #         cache_key = (query, dataset_name, frozenset(matched_ids))
    #         if cache_key in self.bm25_score_cache:
    #             scores = self.bm25_score_cache[cache_key]
    #         else:
    #             scores = bm25.get_batch_scores(tokens, filtered_indices)
    #             self.bm25_score_cache[cache_key] = scores

    #         filtered_ids = [doc_ids[i] for i in filtered_indices]
    #         filtered_texts = [doc_texts[i] for i in filtered_indices]
    #     else:
    #         scores = bm25.get_scores(tokens)
    #         filtered_ids = doc_ids
    #         filtered_texts = doc_texts

    #     if len(scores) == 0:
    #         return []

    #     # Use numpy for efficient top-k selection
    #     scores_array = np.array(scores)
    #     if len(scores_array) <= top_k:
    #         top_indices = np.argsort(-scores_array)
    #     else:
    #         top_indices = np.argpartition(scores_array, -top_k)[-top_k:]
    #         top_indices = top_indices[np.argsort(-scores_array[top_indices])]

    #     return [
    #         {"doc_id": filtered_ids[i], "text": filtered_texts[i], "score": float(scores[i])}
    #         for i in top_indices
    #     ]
    

    def search_bm25(self, query: str, dataset_name: str, top_k=5, with_index=False):
        bm25, doc_ids, doc_texts = self.load_bm25_assets(dataset_name)
        tokens = self.processor.normalize(query)

        if with_index:
            inverted_index = self.load_inverted_index(dataset_name)
            matched_ids = set()
            for token in tokens:
                matched_ids.update(inverted_index.get(token, []))

            if not matched_ids:
                return []

            # Map matched doc_ids to their indices for fast scoring
            id_to_index = {doc_id: i for i, doc_id in enumerate(doc_ids)}
            filtered_indices = [id_to_index[doc_id] for doc_id in matched_ids if doc_id in id_to_index]

            if not filtered_indices:
                return []

            cache_key = (query, dataset_name, frozenset(matched_ids))
            if cache_key in self.bm25_score_cache:
                scores = self.bm25_score_cache[cache_key]
            else:
                # Efficient batch scoring only on filtered indices
                scores = bm25.get_batch_scores(tokens, filtered_indices)
                self.bm25_score_cache[cache_key] = scores

            filtered_ids = [doc_ids[i] for i in filtered_indices]
            filtered_texts = [doc_texts[i] for i in filtered_indices]

        else:
            scores = bm25.get_scores(tokens)
            filtered_ids = doc_ids
            filtered_texts = doc_texts

        if len(scores) == 0:
            return []

        scores_array = np.array(scores)

        if len(scores_array) <= top_k:
            top_indices = np.argsort(-scores_array)
        else:
            # Get top_k indices with highest scores efficiently
            top_indices = np.argpartition(scores_array, -top_k)[-top_k:]
            # Sort those top_k indices by score descending
            top_indices = top_indices[np.argsort(-scores_array[top_indices])]

        return [
            {"doc_id": filtered_ids[i], "text": filtered_texts[i], "score": float(scores_array[i])}
            for i in top_indices
        ]




    # def filter_documents_by_inverted_index(self, query: str, dataset_name: str, doc_ids, doc_texts, matrix):
    #     inverted_index = self.load_inverted_index(dataset_name)
    #     tokens = set(self.processor.normalize(query))

    #     matched_ids = set()
    #     for token in tokens:
    #         matched_ids.update(inverted_index.get(token, []))

    #     # Filter everything by matched doc_ids
    #     filtered_data = [(i, doc_id, text)
    #                      for i, (doc_id, text) in enumerate(zip(doc_ids, doc_texts))
    #                      if doc_id in matched_ids]

    #     if not filtered_data:
    #         return [], [], [], None

    #     indices, filtered_ids, filtered_texts = zip(*filtered_data)
    #     filtered_matrix = matrix[list(indices)]

    #     return list(filtered_ids), list(filtered_texts), list(indices), filtered_matrix


    

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

    # def search_word2vec(self, query: str, dataset_name: str, top_k=5, with_index=False):
    #     self.load_word2vec_assets(dataset_name)

    #     # if with_index:
    #     #     doc_ids, doc_texts, indices, matrix = self.filter_documents_by_inverted_index(
    #     #         query, dataset_name, self.doc_ids, self.doc_texts, self.w2v_matrix)
    #     #     if not doc_ids:
    #     #         return []
    #     # else:
    #     doc_ids, doc_texts, matrix = self.doc_ids, self.doc_texts, self.w2v_matrix

    #     tokens = self.processor.normalize(query)
    #     vector = np.mean(
    #         [self.w2v_model.wv[w] for w in tokens if w in self.w2v_model.wv]
    #         or [np.zeros(self.w2v_model.vector_size)],
    #         axis=0,
    #     )

    #     similarities = cosine_similarity([vector], matrix).flatten()
    #     top_indices = similarities.argsort()[::-1][:top_k]

    #     return [
    #         {"doc_id": doc_ids[i], "text": doc_texts[i], "score": float(similarities[i])}
    #         for i in top_indices
    #     ]

    # def search_word2vec(self, query: str, dataset_name: str, top_k=5, with_additional=False):
    #     self.load_word2vec_assets(dataset_name)
    #     tokens = self.processor.normalize(query)
        
    #     query_vector = np.mean(
    #         [self.w2v_model.wv[w] for w in tokens if w in self.w2v_model.wv]
    #         or [np.zeros(self.w2v_model.vector_size)],
    #         axis=0,
    #     ).astype("float32")

    #     # if with_additional:
    #     #     # Use FAISS
    #     #     index, doc_ids, doc_texts = self.load_faiss_assets(dataset_name)
    #     #     distances, neighbors = index.search(np.array([query_vector]), top_k)
    #     if with_additional:
    #         index, doc_ids, doc_texts = self.load_faiss_assets(dataset_name)
    #         faiss.normalize_L2(query_vector.reshape(1, -1))
    #         distances, neighbors = index.search(np.array([query_vector]), top_k)


    #         return[
    #             {
    #                 "doc_id": doc_ids[i],
    #                 "text": doc_texts[i],
    #                 # "distance": float(distances[0][idx]),
    #                 # "score": float(1 / (1 + distances[0][idx]))  # optional conversion to similarity
    #                  "score": float(distances[0][idx])
    #             }
    #             for idx, i in enumerate(neighbors[0])
    #         ]
    #     else:
    #         # Use cosine similarity with raw matrix
    #         doc_ids, doc_texts, matrix = self.doc_ids, self.doc_texts, self.w2v_matrix
    #         similarities = cosine_similarity([query_vector], matrix).flatten()
    #         top_indices = similarities.argsort()[::-1][:top_k]

    #         return [
    #             {"doc_id": doc_ids[i], "text": doc_texts[i], "score": float(similarities[i])}
    #             for i in top_indices
    #         ]


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
        # Use FAISS for fast similarity search
            index, doc_ids, doc_texts = self.load_faiss_assets(dataset_name)

            # Normalize query vector for cosine similarity (inner product)
            faiss.normalize_L2(query_vector)
            
            # Search top_k similar documents
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


        # query_vector = self.word2vecRep.vectorize(tokens)


        # print(query_vector)

        # results = self.w2v_model.wv.most_similiar(tokens , topn=10)

        # return results
        # query_vector = np.mean(
        #     [self.w2v_model.wv[w] for w in tokens if w in self.w2v_model.wv]
        #     or [np.zeros(self.w2v_model.vector_size)],
        #     axis=0,
        # ).astype("float32")

        # if with_additional:
        #     index, doc_ids, doc_texts = self.load_faiss_assets(dataset_name)
        #     faiss.normalize_L2(query_vector.reshape(1, -1))
        #     distances, neighbors = index.search(np.array([query_vector]), top_k)

        #     return [
        #         {
        #             "doc_id": doc_ids[i],
        #             "text": doc_texts[i],
        #             "score": float(distances[0][idx])
        #         }
        #         for idx, i in enumerate(neighbors[0])
        #     ]
        # else:
        #     doc_ids, doc_texts, matrix = self.doc_ids, self.doc_texts, self.w2v_matrix

        #     # Normalize query_vector and matrix rows
        #     query_norm = query_vector / (np.linalg.norm(query_vector) + 1e-10)
        #     matrix_norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
        #     matrix_normalized = matrix / matrix_norms

        #     similarities = np.dot(matrix_normalized, query_norm)

        #     top_indices = np.argpartition(-similarities, top_k - 1)[:top_k]
        #     top_indices = top_indices[np.argsort(-similarities[top_indices])]  # sort top_k results

        #     return [
        #         {"doc_id": doc_ids[i], "text": doc_texts[i], "score": float(similarities[i])}
        #         for i in top_indices
        #     ]

    # def search_hybrid(self, query: str, dataset_name: str, top_k=5, alpha=0.5, with_index=False):
    #     vectorizer, tfidf_matrix, doc_ids, doc_texts = self.load_tfidf_assets(dataset_name)
    #     self.load_word2vec_assets(dataset_name)

    #     if with_index:
    #         doc_ids, doc_texts, indices, tfidf_matrix = self.filter_documents_by_inverted_index(
    #             query, dataset_name, doc_ids, doc_texts, tfidf_matrix)
    #         if not doc_ids:
    #             return []

    #         w2v_matrix = self.w2v_matrix[indices]
    #     else:
    #         w2v_matrix = self.w2v_matrix

    #     tokens = self.processor.normalize(query)
    #     tfidf_vector = vectorizer.transform([" ".join(tokens)])
    #     w2v_vector = np.mean(
    #         [self.w2v_model.wv[w] for w in tokens if w in self.w2v_model.wv]
    #         or [np.zeros(self.w2v_model.vector_size)],
    #         axis=0,
    #     )

    #     sim_tfidf = cosine_similarity(tfidf_vector, tfidf_matrix).flatten()
    #     sim_w2v = cosine_similarity([w2v_vector], w2v_matrix).flatten()

    #     final_scores = alpha * sim_tfidf + (1 - alpha) * sim_w2v
    #     top_indices = final_scores.argsort()[::-1][:top_k]

    #     return [
    #         {"doc_id": doc_ids[i], "text": doc_texts[i], "score": float(final_scores[i])}
    #         for i in top_indices
    #     ]
    def search_hybrid(self, query: str, dataset_name: str, top_k=5, alpha=0.4, beta=0.3, gamma=0.3, with_index=False,with_additional=False):
        # Load all assets
        vectorizer, tfidf_matrix, doc_ids, doc_texts = self.load_tfidf_assets(dataset_name)
        self.load_word2vec_assets(dataset_name)
        bm25, _, _ = self.load_bm25_assets(dataset_name)

        if with_index:
            doc_ids, doc_texts, indices, tfidf_matrix = self.filter_documents_by_inverted_index(
                query, dataset_name, doc_ids, doc_texts, tfidf_matrix)
            if not doc_ids:
                return []

            w2v_matrix = self.w2v_matrix[indices]
            bm25_indices = indices
        else:
            w2v_matrix = self.w2v_matrix
            bm25_indices = list(range(len(doc_ids)))

        # Preprocess query
        tokens = self.processor.normalize(query)

        # TF-IDF vector + sim
        tfidf_vector = vectorizer.transform([" ".join(tokens)])
        sim_tfidf = cosine_similarity(tfidf_vector, tfidf_matrix).flatten()
        sim_tfidf /= sim_tfidf.max() or 1

        # # Word2Vec vector + sim
        # w2v_vector = np.mean(
        #     [self.w2v_model.wv[w] for w in tokens if w in self.w2v_model.wv]
        #     or [np.zeros(self.w2v_model.vector_size)],
        #     axis=0
        # )
        # sim_w2v = cosine_similarity([w2v_vector], w2v_matrix).flatten()
        # sim_w2v /= sim_w2v.max() or 1

           # --- Word2Vec embedding ---
        w2v_query_vec = np.mean(
            [self.w2v_model.wv[w] for w in tokens if w in self.w2v_model.wv]
            or [np.zeros(self.w2v_model.vector_size)],
            axis=0
        ).astype("float32")

        # if with_additional:
        #     # FAISS search on Word2Vec
        #     index, faiss_doc_ids, faiss_doc_texts = self.load_faiss_assets(dataset_name)
        #     distances, neighbors = index.search(np.array([w2v_query_vec]), len(faiss_doc_ids))
        #     sim_w2v = 1 / (1 + distances[0])  # Convert distances to similarity
        #     sim_w2v /= sim_w2v.max() or 1

        #     # Align doc_ids order with other scores
        #     faiss_index_to_id = {doc_id: i for i, doc_id in enumerate(faiss_doc_ids)}
        #     ordered_indices = [faiss_index_to_id.get(doc_id, -1) for doc_id in doc_ids]

        #     # If index not found, use 0 similarity
        #     sim_w2v = np.array([sim_w2v[i] if i != -1 else 0 for i in ordered_indices])
        # else:
        #     # Regular Word2Vec cosine sim
        #     w2v_matrix = self.w2v_matrix if not with_index else self.w2v_matrix[w2v_indices]
        #     sim_w2v = cosine_similarity([w2v_query_vec], w2v_matrix).flatten()
        #     sim_w2v /= sim_w2v.max() or 1
        if with_additional:
            # FAISS cosine sim (requires normalized vectors and IndexFlatIP)
            faiss.normalize_L2(w2v_query_vec.reshape(1, -1))
            index, faiss_doc_ids, faiss_doc_texts = self.load_faiss_assets(dataset_name)
            distances, neighbors = index.search(w2v_query_vec.reshape(1, -1), len(faiss_doc_ids))
            sim_w2v_faiss = distances[0]  # Already cosine similarity (dot product)

            # Align sim_w2v with doc_ids
            id_to_score = {doc_id: score for doc_id, score in zip(faiss_doc_ids, sim_w2v_faiss)}
            sim_w2v = np.array([id_to_score.get(doc_id, 0) for doc_id in doc_ids])
            sim_w2v /= sim_w2v.max() or 1
        else:
            sim_w2v = cosine_similarity([w2v_query_vec], w2v_matrix).flatten()
            sim_w2v /= sim_w2v.max() or 1


        # BM25 score
        scores_bm25 = bm25.get_batch_scores(tokens, bm25_indices)
        scores_bm25 = np.array(scores_bm25)
        scores_bm25 /= scores_bm25.max() or 1

        # Combine all three
        final_scores = (
            alpha * sim_tfidf +
            beta * sim_w2v +
            gamma * scores_bm25
        )

        # Sort and return top_k
        top_indices = final_scores.argsort()[::-1][:top_k]

        return [
            {
                "doc_id": doc_ids[i],
                "text": doc_texts[i],
                "score": float(final_scores[i])
            }
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
        # elif algorithm == "faiss":
        #     return self.search_faiss(query, dataset_name, top_k, with_index)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
