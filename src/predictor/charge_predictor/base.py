from abc import ABC, abstractmethod


class BaseChargePredictor(ABC):
    @abstractmethod
    def predict(self, fact: str, defendants: list[str]) -> dict[str, list[str]]:
        """
        Predicts the charge based on the provided fact and defendants.

        Args:
            fact (str): The fact to analyze.
            defendants (list[str]): List of defendants involved in the case.

        Returns:
            dict: A dictionary containing the predicted charge and its details.
        """
