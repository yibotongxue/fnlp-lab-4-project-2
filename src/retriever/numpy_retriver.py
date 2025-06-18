from typing import Callable, override

import numpy as np

from .vector import VectorRetriever
from ..embed import BaseEmbedding


class NumpyRetriever(VectorRetriever):
    def __init__(
        self,
        embedder: BaseEmbedding,
        text_getter: (
            Callable[[int], str]
            | Callable[[list[int]], list[str]]
            | Callable[[list[list[int]]], list[list[str]]]
        ),
        embedding_matrix: np.ndarray,
    ):
        super().__init__(embedder, text_getter)
        self.embedding_matrix = embedding_matrix
        self.embedding_matrix_normal = np.linalg.norm(self.embedding_matrix, axis=1)

    @override
    def get_most_similar_ids(
        self, query_embedding: np.ndarray, k: int
    ) -> tuple[list[int], list[float]]:
        query_embedding_normal = np.linalg.norm(query_embedding)
        similarity_scores = np.dot(self.embedding_matrix, query_embedding) / (
            self.embedding_matrix_normal * query_embedding_normal
        )
        indices = np.argpartition(similarity_scores, -k)[-k:]
        sorted_indices = indices[np.argsort(-similarity_scores[indices])].tolist()
        sorted_scores = similarity_scores[sorted_indices].tolist()
        return sorted_indices, sorted_scores
