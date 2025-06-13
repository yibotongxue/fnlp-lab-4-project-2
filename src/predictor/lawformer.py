from typing import override

import numpy as np
import torch
from transformers import AutoTokenizer

from .base import BasePredictor
from ..utils.type_utils import OutcomeDict
from ..finetune.model import LegalPredictionModel


class LawformerPredictor(BasePredictor):
    def __init__(
        self,
        model_path: str,
        base_model_name: str,
        charge_num: int = 321,
        imprisonment_num: int = 600,
        device: torch.device = torch.device("cpu"),
        charge_id_mapping: dict = None,
    ):
        """
        Initializes the LawformerPredictor with a pre-trained model.

        Args:
            model_path (str): Path to the pre-trained model.
            base_model_name (str): Name of the base model.
            charge_num (int): Number of charge classes.
            imprisonment_num (int): Number of imprisonment classes.
            device (torch.device): Device to run the model on (default is CPU).
        """
        super().__init__()
        self.model = LegalPredictionModel.from_pretrained(
            safetensors_path=f"{model_path}/model.safetensors",
            base_model_name=base_model_name,
            charge_num=charge_num,
            imprisonment_num=imprisonment_num,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.device = device
        self.charge_id_mapping = charge_id_mapping if charge_id_mapping else {}
        self.model.eval()
        self.model.to(self.device)

    @override
    def predict_judgment(self, fact: str, defendants: list[str]) -> list[OutcomeDict]:
        """
        Predicts the standard accusation judgment based on the provided fact.

        Args:
            fact (str): The fact to analyze.
            defendants (list[str]): List of defendants.

        Returns:
            list[OutcomeDict]: A list of predicted outcomes, each containing the name and judgment details.
        """

        # Process logits to create OutcomeDicts
        outcomes = []
        for _, defendant in enumerate(defendants):
            inputs = self.tokenizer(
                f"【当前被告人：{defendant}】" + fact,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)

            charge_logits = outputs.charge_logits.cpu().numpy()
            imprisonment_logits = outputs.imprisonment_logits.cpu().numpy()
            charge_id = np.argmax(charge_logits, axis=1)
            imprisonment = np.max(imprisonment_logits, axis=1).astype(int)
            outcome = {
                "name": defendant,
                "judgment": [
                    {
                        "standard_accusation": self.charge_id_mapping.get(
                            charge_id[0], "Unknown Charge"
                        ),
                        "imprisonment": int(imprisonment[0]),
                    }
                ],
            }
            outcomes.append(OutcomeDict(**outcome))

        return outcomes


if __name__ == "__main__":
    import argparse
    import os
    from ..utils.json_util import save_json
    from ..utils.data_utils import LegalCaseDataset, ChargeLoader

    parser = argparse.ArgumentParser(description="Lawformer Predictor")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the pre-trained Lawformer model",
    )
    parser.add_argument(
        "--base-model-name",
        type=str,
        default="bert-base-uncased",
        help="Name of the base model to use",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test"],
        help="Dataset split to use (train or test)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Directory to save the output results",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Index to start processing cases from",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=None,
        help="Number of training cases to use for zero-shot prediction",
    )
    parser.add_argument(
        "--charge-file",
        type=str,
        default="./data/charges.json",
        help="Path to the charge list file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run the model on (cpu or cuda)",
    )
    args = parser.parse_args()
    model_path = args.model_path
    base_model_name = args.base_model_name
    split = args.split
    output_dir = args.output_dir
    start_index = args.start_index
    train_size = args.train_size
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    def get_data(split: str = "train") -> LegalCaseDataset:
        """Load the legal case dataset."""
        if split == "train":
            return LegalCaseDataset("./data/train.jsonl")
        elif split == "test":
            return LegalCaseDataset("./data/test.jsonl")
        else:
            raise ValueError("Invalid split. Use 'train' or 'test'.")

    legal_data = get_data(split)
    if train_size is not None:
        legal_data = legal_data[start_index : start_index + train_size]
    else:
        legal_data = legal_data[start_index:]
    charge_loader = ChargeLoader(args.charge_file)
    predictor = LawformerPredictor(
        model_path,
        base_model_name,
        charge_id_mapping=charge_loader.reverse_charges,
        device=device,
    )
    for i, data in enumerate(legal_data):
        result_to_save = {}
        result_to_save["input"] = data.model_dump()
        result = predictor.predict_judgment(fact=data.fact, defendants=data.defendants)
        result_to_save["result"] = [outcome.model_dump() for outcome in result]
        save_json(result_to_save, os.path.join(output_dir, f"{i + start_index}.json"))
        print(f"Processed case {i + 1}/{len(legal_data)}")
    print(f"Results saved to {output_dir}")
