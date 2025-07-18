from typing import override

from .base import BasePredictor
from .charge_predictor import BaseChargePredictor
from .imprisonment_predictor import BaseImprisonmentPredictor
from ..utils.data_utils import OutcomeDict


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
    from tqdm import tqdm

    from .charge_predictor import get_charge_predictor
    from .imprisonment_predictor import get_imprisonment_predictor
    from ..utils import save_json
    from ..utils.data_utils import LegalCaseDataset

    load_dotenv()

    def _get_data(data_dir: str, split: str = "train") -> LegalCaseDataset:
        """Load the legal case dataset."""
        return LegalCaseDataset(os.path.join(data_dir, f"{split}.jsonl"))

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
        "--charge-file",
        type=str,
        default="./data/charges.json",
        help="Path to the charge list file",
    )
    parser.add_argument(
        "--multiple-predictor-type",
        type=str,
        required=False,
        help="Type of multiple predictor",
    )
    parser.add_argument(
        "--candidate-cnt", type=int, required=False, help="Count of candidates"
    )
    parser.add_argument(
        "--multiple-charge-model-path",
        type=str,
        required=False,
        help="Path to the pre-trained Lawformer model",
    )
    parser.add_argument(
        "--multiple-device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run the model on (cpu or cuda)",
    )
    parser.add_argument(
        "--refiner-type", type=str, required=False, help="Type of refiner"
    )
    parser.add_argument(
        "--refiner-llm", type=str, required=False, help="LLM for refiner"
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
        "--imprisonment-mapper-config",
        type=str,
        required=False,
        help="Path of the config file of imprisonment mapper",
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
        "--data-dir",
        type=str,
        default="./data",
        help="The path to the directory of the data, which could contains the file train.jsonl and test.jsonl",
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
    legal_data = _get_data(args.data_dir, args.split)
    if args.train_size is not None:
        legal_data = legal_data[args.start_index : args.start_index + args.train_size]
    else:
        legal_data = legal_data[args.start_index :]
    for i in tqdm(range(len(legal_data)), "Legal Judgement Predition"):
        data = legal_data[i]
        result_to_save = {}
        result_to_save["input"] = data.model_dump()
        result = predictor.predict_judgment(fact=data.fact, defendants=data.defendants)
        result_to_save["result"] = [outcome.model_dump() for outcome in result]
        save_json(
            result_to_save,
            os.path.join(args.output_dir, f"{i + args.start_index}.json"),
        )
    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
