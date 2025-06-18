from openai import OpenAI

from .base import BaseEmbedding


class _BaseOpenAIEmbedding(BaseEmbedding):
    def __init__(
        self,
        base_url: str,
        model_name: str,
        api_key: str = None,
    ):
        """
        Initialize the OpenAI embedding client.

        Args:
            base_url (str): The base URL for the OpenAI API.
            api_key (str): The API key for authentication.
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

    def embed(self, text: str) -> list[float]:
        """
        Embed a single text string into a vector.

        Args:
            text (str): The text to embed.

        Returns:
            list[float]: The embedded vector.
        """
        response = self.client.embeddings.create(
            model=self.model_name, input=text, encoding_format="float"
        )
        return response.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of text strings into vectors.

        Args:
            texts (list[str]): The texts to embed.

        Returns:
            list[list[float]]: The embedded vectors.
        """
        if not all(isinstance(text, str) for text in texts):
            raise ValueError("Expected a list of strings, but got non-string elements")

        response = self.client.embeddings.create(
            model="text-embedding-v4", input=texts, encoding_format="float"
        )
        return [item.embedding for item in response.data]


class QwenTextEmbedding(_BaseOpenAIEmbedding):
    BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    VALID_MODELS: set[str] = {
        "text-embedding-v4",
        "text-embedding-v3",
        "text-embedding-v2",
        "text-embedding-v1",
    }
    DIMENSION_DICT: dict[str, set[int]] = {
        "text-embedding-v4": {2048, 1536, 1024, 768, 512, 256, 128, 64},
        "text-embedding-v3": {1024, 768, 512, 256, 128, 64},
    }

    def __init__(self, model_name: str, api_key: str = None, dimension: int = None):
        super().__init__(base_url=self.BASE_URL, model_name=model_name, api_key=api_key)
        assert (
            model_name in self.VALID_MODELS
        ), f"Invalid model name: {model_name}. Valid models are: {self.VALID_MODELS}"
        if model_name not in self.DIMENSION_DICT:
            assert (
                dimension is None
            ), f"Model {model_name} does not support specifying dimension."
        elif dimension is not None:
            assert (
                dimension in self.DIMENSION_DICT[model_name]
            ), f"Model {model_name} does not support dimension {dimension}. Supported dimensions are: {self.DIMENSION_DICT[model_name]}"
            self.dimension = dimension
        else:
            self.dimension = 1024

    def embed(self, text: str) -> list[float]:
        """
        Embed a single text string into a vector using the Qwen model.

        Args:
            text (str): The text to embed.

        Returns:
            list[float]: The embedded vector.
        """
        if self.model_name in self.DIMENSION_DICT:
            response = self.client.embeddings.create(
                model=self.model_name,
                input=text,
                encoding_format="float",
                dimensions=self.dimension,
            )
        else:
            response = self.client.embeddings.create(
                model=self.model_name, input=text, encoding_format="float"
            )
        return response.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of text strings into vectors using the Qwen model.

        Args:
            texts (list[str]): The texts to embed.

        Returns:
            list[list[float]]: The embedded vectors.
        """
        if self.model_name in self.DIMENSION_DICT:
            response = self.client.embeddings.create(
                model=self.model_name,
                input=texts,
                encoding_format="float",
                dimensions=self.dimension,
            )
        else:
            response = self.client.embeddings.create(
                model=self.model_name, input=texts, encoding_format="float"
            )
        return [item.embedding for item in response.data]
