from abc import ABC, abstractmethod
from typing import Any


class BaseTrainer(ABC):
    def __init__(
        self,
        cfgs: dict[str, Any],
    ):
        self.cfgs = cfgs
        print(f"Training configuration: {self.cfgs}")
        self.init_model()
        print(f"Model initialized: {self.model.__class__.__name__}")
        self.init_datasets()
        print("Datasets initialized.")
        self.init_trainer()
        print("Trainer initialized.")

    @abstractmethod
    def init_model(self):
        """Initialize the model."""

    @abstractmethod
    def init_datasets(self):
        """Initialize the datasets."""

    @abstractmethod
    def init_trainer(self):
        """Initialize the training arguments."""

    @abstractmethod
    def compute_metrics(self, eval_pred):
        """
        Compute metrics based on the evaluation predictions.

        Args:
            eval_pred: The evaluation predictions.

        Returns:
            dict: A dictionary containing the computed metrics.
        """

    def train(self):
        """Train the model."""
        self.trainer.train()

    def eval(self) -> dict[str, float]:
        """Evaluate the model."""
        return self.trainer.evaluate()
