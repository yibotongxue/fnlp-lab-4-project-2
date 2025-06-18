from abc import ABC, abstractmethod

import torch
import numpy as np


class BaseEmbedding(ABC):
    """
    Base class for all embedder classes.
    """

    @abstractmethod
    def embed(self, text: str) -> list[float] | np.ndarray | torch.Tensor:
        """
        Embed a single text string into a vector.

        Args:
            text (str): The text to embed.

        Returns:
            list[float] | np.ndarray | torch.Tensor: The embedded vector.
        """

    def embed_batch(
        self, texts: list[str]
    ) -> list[list[float]] | list[np.ndarray] | list[torch.Tensor]:
        """
        Embed a batch of text strings into vectors.

        Args:
            texts (list[str]): The texts to embed.

        Returns:
            list[list[float]] | list[np.ndarray] | list[torch.Tensor]: The embedded vectors.
        """
        for text in texts:
            if not isinstance(text, str):
                raise ValueError(f"Expected a list of strings, but got {type(text)}")
        return [self.embed(text) for text in texts]
