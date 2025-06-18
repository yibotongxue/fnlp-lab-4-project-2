from abc import abstractmethod
from typing import Callable, override

import numpy as np

from .base import BaseRetriever
from ..embed import BaseEmbedding


class VectorRetriever(BaseRetriever):
    def __init__(
        self,
        embedder: BaseEmbedding,
        text_getter: (
            Callable[[int], str]
            | Callable[[list[int]], list[str]]
            | Callable[[list[list[int]]], list[list[str]]]
        ),
    ):
        self.embedder = embedder
        self.text_getter = text_getter

    @override
    def retrieve(self, query: str, k: int = 3) -> list[str]:
        query_embedding = np.array(self.embedder.embed(query))
        ids: list[int] = self.get_most_similar_ids(query_embedding, k)[0]
        return self.text_getter(ids)

    @override
    def retrieve_batch(self, queries: list[str], k: int = 3) -> list[list[str]]:
        query_embedding = np.array(self.embedder.embed_batch(queries))
        ids: list[list[int]] = self.get_most_similar_ids_batch(query_embedding, k)[0]
        return self.text_getter(ids)

    @abstractmethod
    def get_most_similar_ids(
        self, query_embedding: np.ndarray, k: int
    ) -> tuple[list[int], list[float]]:
        pass

    def get_most_similar_ids_batch(
        self, query_embeddings: list[np.ndarray], k: int
    ) -> tuple[list[list[int]], list[list[float]]]:
        ids = []
        scores = []
        for query_embedding in query_embeddings:
            ids.append(self.get_most_similar_ids(query_embedding, k))
            scores.append(self.get_most_similar_ids(query_embedding, k))
        return ids, scores
