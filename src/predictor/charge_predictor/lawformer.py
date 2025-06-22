from typing import override

import torch

from .base import BaseChargePredictor
from .multiple_predictor import LawformerMultipleChargePredictor


class LawformerChargePredictor(BaseChargePredictor):
    def __init__(
        self,
        charge_model_path: str,
        charge_num: int = 321,
        device: torch.device = torch.device("cpu"),
        charge_id_mapping: dict = None,
    ):
        """
        Initializes the LawformerChargePredictor with a pre-trained model.
        Args:
            charge_model_path (str): Path to the pre-trained charge prediction model.
            base_model_name (str): Name of the base model.
            charge_num (int): Number of charge classes.
            device (torch.device): Device to run the model on (default is CPU).
            charge_id_mapping (dict, optional): Mapping from charge IDs to human-readable names.
        """
        super().__init__()
        self.multiple_predictor = LawformerMultipleChargePredictor(
            candidate_cnt=1,
            charge_model_path=charge_model_path,
            charge_num=charge_num,
            device=device,
            charge_id_mapping=charge_id_mapping,
        )

    @override
    def predict(self, fact: str, defendants: list[str]) -> dict[str, list[str]]:
        """Predict the charges for each defendant based on the provided fact.
        Args:
            fact (str): The fact to analyze.
            defendants (list[str]): List of defendants involved in the case.
        Returns:
            dict: A dictionary where keys are defendant names and values are lists of predicted charges.
        """
        result = self.multiple_predictor.predict(fact, defendants)
        for key in result.keys():
            result[key] = result[key][0]
        return result
