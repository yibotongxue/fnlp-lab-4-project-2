from abc import ABC, abstractmethod

from ..utils.data_utils import OutcomeDict


class BasePredictor(ABC):
    """
    Abstract base class for all predictors.
    """

    @abstractmethod
    def predict_judgment(self, fact: str, defendants: list[str]) -> list[OutcomeDict]:
        """
        Predicts the standard accusation judgment based on the provided fact.

        Args:
            fact (str): The fact to analyze.

        Returns:
            list[OutcomeDict]: A list of predicted outcomes, each containing the name and judgment details.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
