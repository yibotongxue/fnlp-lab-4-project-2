from typing import override

import numpy as np
import torch

from .base import BaseMultipleChargePredictor
from ....finetune.utils import load_pretrained_models


class LawformerMultipleChargePredictor(BaseMultipleChargePredictor):
    def __init__(
        self,
        candidate_cnt: int,
        charge_model_path: str,
        charge_num: int = 321,
        device: torch.device = torch.device("cpu"),
        charge_id_mapping: dict = None,
    ):
        """
        Initializes the LawformerChargePredictor with a pre-trained model.
        Args:
            candidate_cnt (int): Count of candidates
            charge_model_path (str): Path to the pre-trained charge prediction model.
            charge_num (int): Number of charge classes.
            device (torch.device): Device to run the model on (default is CPU).
            charge_id_mapping (dict, optional): Mapping from charge IDs to human-readable names.
        """
        super().__init__(candidate_cnt)
        self.charge_model, self.charge_tokenizer = load_pretrained_models(
            charge_model_path,
            is_classification=True,
            auto_model_kwargs={
                "num_labels": charge_num,
            },
        )
        self.device = device
        self.charge_id_mapping = charge_id_mapping if charge_id_mapping else {}
        self.charge_model.eval()
        self.charge_model.to(self.device)

    @override
    def predict(self, fact: str, defendants: list[str]) -> dict[str, list[list[str]]]:
        """Predict the charges for each defendant based on the provided fact.
        Args:
            fact (str): The fact to analyze.
            defendants (list[str]): List of defendants involved in the case.
        Returns:
            dict: A dictionary where keys are defendant names and values are lists of predicted charges.
        """
        result = {}
        for _, defendant in enumerate(defendants):
            inputs = self.charge_tokenizer(
                f"【当前被告人：{defendant}】" + fact,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)
            with torch.no_grad():
                outputs = self.charge_model(**inputs)
            charge_logits = outputs.cpu().numpy().reshape(-1)
            charge_ids = np.argsort(charge_logits)[-self.candidate_cnt :]
            assert np.argsort(charge_logits)[-1] == np.argmax(
                charge_logits
            ), f"{charge_logits[np.argpartition(charge_logits, 1)[-1]]} != {charge_logits[np.argmax(charge_logits)]}"
            result[defendant] = [
                [
                    self.charge_id_mapping.get(charge_id, "Unknown Charge")
                    for charge_id in charge_ids
                ]
            ]
        return result
