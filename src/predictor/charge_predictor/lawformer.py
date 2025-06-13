from typing import override

import numpy as np
import torch
from transformers import AutoTokenizer

from .base import BaseChargePredictor
from ...finetune.model import LegalSinglePredictionModel


class LawformerChargePredictor(BaseChargePredictor):
    def __init__(
        self,
        charge_model_path: str,
        base_model_name: str,
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
        self.charge_model = LegalSinglePredictionModel.from_pretrained(
            safetensors_path=f"{charge_model_path}/model.safetensors",
            base_model_name=base_model_name,
            num_classes=charge_num,
        )
        self.charge_tokenizer = AutoTokenizer.from_pretrained(
            charge_model_path, use_fast=True
        )
        self.device = device
        self.charge_id_mapping = charge_id_mapping if charge_id_mapping else {}
        self.charge_model.eval()
        self.charge_model.to(self.device)

    @override
    def predict(self, fact: str, defendants: list[str]) -> dict[str, list[str]]:
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

            charge_logits = outputs.cpu().numpy()
            charge_id = np.argmax(charge_logits, axis=1)
            charge_name = self.charge_id_mapping.get(charge_id[0], "Unknown Charge")
            result[defendant] = [charge_name]
        return result
