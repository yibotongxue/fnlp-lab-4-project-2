import os

from .base import BaseEmbedding
from .openai import QwenTextEmbedding


def get_embedding_model(
    model_name: str = "text-embedding-v3",
    api_key: str | None = None,
    dimension: int | None = None,
) -> BaseEmbedding:
    if model_name in QwenTextEmbedding.VALID_MODELS:
        if api_key is None:
            api_key = os.getenv("QWEN_API_KEY")
        return QwenTextEmbedding(model_name, api_key=api_key, dimension=dimension)
    else:
        raise ValueError(
            f"Unsupported model name: {model_name}. Supported models are: {QwenTextEmbedding.VALID_MODELS}."
        )
