from abc import ABC, abstractmethod

from ..utils.type_utils import OutcomeDict


class BasePredictor(ABC):
    """
    Abstract base class for all predictors.
    """

    @abstractmethod
    def predict_judgment(self, fact: str, defendants: list[str]) -> OutcomeDict:
        """
        Predicts the standard accusation judgment based on the provided fact.

        Args:
            fact (str): The fact to analyze.

        Returns:
            list[tuple[str, int]]: A list of tuples containing the accusation and its score.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
