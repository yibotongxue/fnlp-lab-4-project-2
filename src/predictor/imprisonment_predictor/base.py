from abc import ABC, abstractmethod


class BaseImprisonmentPredictor(ABC):
    @abstractmethod
    def predict(
        self, fact: str, defendants: list[str], charge_dict: dict[str, list[str]]
    ) -> dict:
        """
        Predicts the imprisonment based on the provided fact, defendants, and charge details.

        Args:
            fact (str): The fact to analyze.
            defendants (list[str]): List of defendants involved in the case.
            charge_dict (dict[str]): Dictionary containing the predicted charge details.

        Returns:
            dict: A dictionary containing the predicted imprisonment and its details.
        """
