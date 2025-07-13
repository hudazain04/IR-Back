from gensim.models import Word2Vec
from typing import List
import numpy as np 

class Word2VecRepresentation:
    def __init__(self, vector_size=100, window=5, min_count=2):
        self.vector_size = vector_size
        self.model = Word2Vec(vector_size=vector_size, window=window, min_count=min_count , workers=4)

    def train(self, tokenized_texts: List[List[str]]):
        self.model.build_vocab(tokenized_texts)
        self.model.train(tokenized_texts, total_examples=self.model.corpus_count, epochs=self.model.epochs)

    def vectorize(self, tokens: List[str]) -> np.ndarray:
        vectors = [self.model.wv[word] for word in tokens if word in self.model.wv]
        if vectors:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(self.vector_size)
