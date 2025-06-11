from typing import override

import numpy as np
import torch
from transformers import AutoTokenizer

from .one_by_one import OneByOnePredictor
from ..finetune.model import LegalSinglePredictionModel


class LawformerOneByOnePredictor(OneByOnePredictor):
    def __init__(
        self,
        charge_model_path: str,
        imprisonment_model_path: str,
        base_model_name: str,
        charge_num: int = 321,
        imprisonment_num: int = 600,
        device: torch.device = torch.device("cpu"),
        charge_id_mapping: dict = None,
    ):
        """
        Initializes the LawformerOneByOnePredictor with a pre-trained model.

        Args:
            charge_model_path (str): Path to the pre-trained charge prediction model.
            imprisonment_model_path (str): Path to the pre-trained imprisonment prediction model.
            base_model_name (str): Name of the base model.
            charge_num (int): Number of charge classes.
            imprisonment_num (int): Number of imprisonment classes.
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
        # self.imprisonment_model = LegalSinglePredictionModel.from_pretrained(
        #     safetensors_path=f"{imprisonment_model_path}/model.safetensors",
        #     base_model_name=base_model_name,
        #     num_classes=imprisonment_num,
        # )
        # self.imprisonment_tokenizer = AutoTokenizer.from_pretrained(imprisonment_model_path, use_fast=True)
        self.device = device
        self.charge_id_mapping = charge_id_mapping if charge_id_mapping else {}
        self.charge_model.eval()
        # self.imprisonment_model.eval()
        self.charge_model.to(self.device)
        # self.imprisonment_model.to(self.device)

    @override
    def predict_charge(self, fact: str, defendants: list[str]) -> dict[str, list[str]]:
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

    @override
    def predict_imprisonment(
        self, fact: str, defendants: list[str], charge_dict: dict[str, list[str]]
    ) -> dict:
        # Implement the logic to predict imprisonment using Lawformer
        result = {}
        for defendant in defendants:
            result[defendant] = []
            for charge in charge_dict.get(defendant, []):
                result[defendant].append(
                    {"standard_accusation": charge, "imprisonment": 0}
                )
        return result


if __name__ == "__main__":
    import argparse
    import os
    from ..utils.json_util import save_json
    from ..utils.data_utils import LegalCaseDataset, ChargeLoader

    parser = argparse.ArgumentParser(description="Lawformer Predictor")
    parser.add_argument(
        "--charge-model-path",
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
    charge_model_path = args.charge_model_path
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
    predictor = LawformerOneByOnePredictor(
        charge_model_path,
        "",
        base_model_name,
        charge_id_mapping=charge_loader.reverse_charges,
        device=device,
    )
    results = []
    for i, data in enumerate(legal_data):
        result_to_save = {}
        result_to_save["input"] = data.model_dump()
        result = predictor.predict_judgment(fact=data.fact, defendants=data.defendants)
        result_to_save["result"] = [outcome.model_dump() for outcome in result]
        save_json(result_to_save, os.path.join(output_dir, f"{i + start_index}.json"))
        print(f"Processed case {i + 1}/{len(legal_data)}")
    print(f"Results saved to {output_dir}")
