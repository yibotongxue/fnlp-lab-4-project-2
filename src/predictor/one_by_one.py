from typing import override

from .base import BasePredictor
from .charge_predictor import BaseChargePredictor
from .imprisonment_predictor import BaseImprisonmentPredictor
from ..utils.type_utils import OutcomeDict


class OneByOnePredictor(BasePredictor):
    def __init__(
        self,
        charge_predictor: BaseChargePredictor,
        imprisonment_predictor: BaseImprisonmentPredictor,
    ):
        """
        Initializes the OneByOnePredictor with the provided charge and imprisonment predictors.

        Args:
            charge_predictor (BaseChargePredictor): The predictor for charges.
            imprisonment_predictor (BaseImprisonmentPredictor): The predictor for imprisonment.
        """
        self.charge_predictor = charge_predictor
        self.imprisonment_predictor = imprisonment_predictor

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
        charge_dict = self.charge_predictor.predict(fact, defendants)
        imprisonment_dict = self.imprisonment_predictor.predict(
            fact, defendants, charge_dict
        )
        return [
            OutcomeDict(name=defendant, judgment=imprisonment_dict[defendant])
            for defendant in defendants
        ]


def main():
    import argparse
    import os

    from dotenv import load_dotenv

    from .charge_predictor import get_charge_predictor
    from .imprisonment_predictor import get_imprisonment_predictor
    from ..utils import save_json
    from ..utils.data_utils import LegalCaseDataset

    load_dotenv()

    def _get_data(split: str = "train") -> LegalCaseDataset:
        """Load the legal case dataset."""
        if split == "train":
            return LegalCaseDataset("./data/train.jsonl")
        elif split == "test":
            return LegalCaseDataset("./data/test.jsonl")
        else:
            raise ValueError("Invalid split. Use 'train' or 'test'.")

    parser = argparse.ArgumentParser(description="One by One Predictor")
    parser.add_argument(
        "--charge-predictor",
        type=str,
        required=True,
        help="Path to the charge predictor model",
    )
    parser.add_argument(
        "--charge-llm",
        type=str,
        default="qwen-max",
        help="LLM model to use for charge prediction",
    )
    parser.add_argument(
        "--charge-model-path",
        type=str,
        required=False,
        help="Path to the pre-trained Lawformer model",
    )
    parser.add_argument(
        "--charge-base-model-name",
        type=str,
        default="bert-base-uncased",
        help="Name of the base model to use",
    )
    parser.add_argument(
        "--charge-file",
        type=str,
        default="./data/charges.json",
        help="Path to the charge list file",
    )
    parser.add_argument(
        "--imprisonment-predictor",
        type=str,
        required=True,
        help="Path to the imprisonment predictor model",
    )
    parser.add_argument(
        "--imprisonment-llm",
        type=str,
        default="qwen-max",
        help="LLM model to use for imprisonment prediction",
    )
    parser.add_argument(
        "--imprisonment-model-path",
        type=str,
        required=False,
        help="Path to the pre-trained Lawformer model for imprisonment prediction",
    )
    parser.add_argument(
        "--imprisonment-base-model-name",
        type=str,
        default="bert-base-uncased",
        help="Name of the base model to use for imprisonment prediction",
    )
    parser.add_argument(
        "--imprisonment-num",
        type=int,
        default=600,
        help="Number of imprisonment classes (default is 600)",
    )
    parser.add_argument(
        "--charge-imprisonment-dict-path",
        type=str,
        default="./data/charge_imprisonment_dict.json",
        help="Path to the charge imprisonment dictionary file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run the model on (cpu or cuda)",
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
    args = parser.parse_args()
    charge_predictor: BaseChargePredictor = get_charge_predictor(
        args.charge_predictor, args
    )
    imprisonment_predictor: BaseImprisonmentPredictor = get_imprisonment_predictor(
        args.imprisonment_predictor, args
    )
    predictor = OneByOnePredictor(charge_predictor, imprisonment_predictor)
    legal_data = _get_data(args.split)
    if args.train_size is not None:
        legal_data = legal_data[args.start_index : args.start_index + args.train_size]
    else:
        legal_data = legal_data[args.start_index :]
    for i, data in enumerate(legal_data):
        result_to_save = {}
        result_to_save["input"] = data.model_dump()
        result = predictor.predict_judgment(fact=data.fact, defendants=data.defendants)
        result_to_save["result"] = [outcome.model_dump() for outcome in result]
        save_json(
            result_to_save,
            os.path.join(args.output_dir, f"{i + args.start_index}.json"),
        )
        print(f"Processed case {i + 1}/{len(legal_data)}")
    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
