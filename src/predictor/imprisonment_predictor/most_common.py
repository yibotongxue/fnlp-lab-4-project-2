from typing import override

from .base import BaseImprisonmentPredictor
from ...utils import load_json


class MostCommonImprisonmentPredictor(BaseImprisonmentPredictor):
    def __init__(self, charge_imprisonment_dict_path: str):
        self.charge_imprisonment_dict = load_json(charge_imprisonment_dict_path)

    @override
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
        result = {}
        for defendant in defendants:
            result[defendant] = []
            for charge in charge_dict.get(defendant, []):
                result[defendant].append(
                    {
                        "standard_accusation": charge,
                        "imprisonment": self.charge_imprisonment_dict.get(charge, 0),
                    }
                )
        return result
