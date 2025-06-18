from typing import Callable

import numpy as np

from .base import BaseRetriever
from .numpy_retriver import NumpyRetriever
from ..embed import BaseEmbedding


def get_retriever(
    retriever_type: str,
    embedder: BaseEmbedding = None,
    text_getter: (
        Callable[[int], str]
        | Callable[[list[int]], list[str]]
        | Callable[[list[list[int]]], list[list[str]]]
    ) = None,
    embedding_matrix: np.ndarray = None,
) -> BaseRetriever:
    supported_types = ["numpy"]
    if retriever_type == "numpy":
        assert (
            embedder is not None
        ), "The embedder parameter is needed by the NumpyRetriever"
        assert (
            text_getter is not None
        ), "The text getter parameter is needed by the NumpyRetriever"
        assert (
            embedding_matrix is not None
        ), "The embedding matrix is needed by the NumpyRetriever"
        return NumpyRetriever(
            embedder=embedder,
            text_getter=text_getter,
            embedding_matrix=embedding_matrix,
        )
    else:
        raise ValueError(
            f"Unsupported retriever type: {retriever_type}. Supported types are: {supported_types}."
        )
