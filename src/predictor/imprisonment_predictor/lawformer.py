from typing import override

import numpy as np
import torch
from transformers import AutoTokenizer

from .base import BaseImprisonmentPredictor
from ...finetune.model import LegalSinglePredictionModel
from ...utils.imprisonment_mapper import (
    BaseImprisonmentMapper,
    IdentityImprisonmentMapper,
)


class LawformerImprisonmentPredictor(BaseImprisonmentPredictor):
    def __init__(
        self,
        imprisonment_model_path: str,
        base_model_name: str,
        imprisonment_num: int = 321,
        device: torch.device = torch.device("cpu"),
        imprisonment_mapper: BaseImprisonmentMapper = None,
    ):
        """
        Initializes the LawformerChargePredictor with a pre-trained model.
        Args:
            imprisonment_model_path (str): Path to the pre-trained imprisonment prediction model.
            base_model_name (str): Name of the base model.
            imprisonment_num (int): Number of imprisonment classes.
            device (torch.device): Device to run the model on (default is CPU).
        """
        super().__init__()
        self.imprisonment_model = LegalSinglePredictionModel.from_pretrained(
            safetensors_path=f"{imprisonment_model_path}/model.safetensors",
            base_model_name=base_model_name,
            num_classes=imprisonment_num,
        )
        self.imprisonment_tokenizer = AutoTokenizer.from_pretrained(
            imprisonment_model_path, use_fast=True
        )
        self.device = device
        self.imprisonment_model.eval()
        self.imprisonment_model.to(self.device)
        if imprisonment_mapper is None:
            self.imprisonment_mapper = IdentityImprisonmentMapper()
        else:
            self.imprisonment_mapper = imprisonment_mapper

    @override
    def predict(
        self, fact: str, defendants: list[str], charge_dict: dict[str, list[str]]
    ) -> dict:
        """Predict the imprisonments for each defendant based on the provided fact.
        Args:
            fact (str): The fact to analyze.
            defendants (list[str]): List of defendants involved in the case.
        Returns:
            dict: A dictionary where keys are defendant names and values are lists of predicted imprisonments.
        """
        result = {}
        for _, defendant in enumerate(defendants):
            inputs = self.imprisonment_tokenizer(
                f"【当前被告人：{defendant}】，【罪名：{charge_dict[defendant]}】"
                + fact,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)
            with torch.no_grad():
                outputs = self.imprisonment_model(**inputs)

            imprisonment_logits = outputs.cpu().numpy()
            imprisonment = np.argmax(imprisonment_logits, axis=1)[0]
            result[defendant] = [
                {
                    "standard_accusation": charge_dict[defendant][0],
                    "imprisonment": self.imprisonment_mapper.label2imprisonment(
                        imprisonment
                    ),
                }
            ]
        return result
