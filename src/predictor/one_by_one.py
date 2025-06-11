from abc import abstractmethod
from typing import override

from .base import BasePredictor
from ..utils.type_utils import OutcomeDict


class OneByOnePredictor(BasePredictor):
    @override
    def predict_judgment(self, fact: str, defendants: list[str]) -> list[OutcomeDict]:
        """
        Predicts the standard accusation judgment based on the provided fact.

        Args:
            fact (str): The fact to analyze.
            defendants (list[str]): List of defendants involved in the case.

        Returns:
            list[OutcomeDict]: A list of predicted outcomes, each containing the name and judgment details.
        """
        charge_dict = self.predict_charge(fact, defendants)
        imprisonment_dict = self.predict_imprisonment(fact, defendants, charge_dict)
        return [
            OutcomeDict(name=defendant, judgment=imprisonment_dict[defendant])
            for defendant in defendants
        ]

    @abstractmethod
    def predict_charge(self, fact: str, defendants: list[str]) -> dict[str, list[str]]:
        """
        Predicts the charge based on the provided fact and defendants.

        Args:
            fact (str): The fact to analyze.
            defendants (list[str]): List of defendants involved in the case.

        Returns:
            dict: A dictionary containing the predicted charge and its details.
        """

    @abstractmethod
    def predict_imprisonment(
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
