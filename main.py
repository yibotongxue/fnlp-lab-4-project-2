import argparse
import os

from dotenv import load_dotenv

from src.llm import get_llm
from src.predictor import ZeroShotPredictor
from src.utils import save_json
from src.utils.data_utils import LegalCaseDataSet

load_dotenv()


def get_data(split: str = "train") -> LegalCaseDataSet:
    """Load the legal case dataset."""
    if split == "train":
        return LegalCaseDataSet("./data/train.jsonl")
    elif split == "test":
        return LegalCaseDataSet("./data/test.jsonl")
    else:
        raise ValueError("Invalid split. Use 'train' or 'test'.")


def main():
    parser = argparse.ArgumentParser(description="Zero-shot legal case prediction")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Directory to save the output results",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Dataset split to use (train or test)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Name of the LLM model to use",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=10,
        help="Number of training cases to use for zero-shot prediction",
    )
    args = parser.parse_args()
    output_dir = args.output_dir
    split = args.split
    model_name = args.model_name
    train_size = args.train_size

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    legal_data = get_data(split)[:train_size]
    print(f"Loaded {len(legal_data)} cases from the dataset.")
    llm = get_llm(model_name=model_name)
    zero_shot_predictor = ZeroShotPredictor(llm)
    for i, data in enumerate(legal_data):
        result_to_save = {}
        result_to_save["input"] = data.model_dump()
        prompt = zero_shot_predictor.build_zero_shot_prompt(
            fact=data.fact, defendants=data.defendants
        )
        result_to_save["system_prompt"] = zero_shot_predictor.system_prompt
        result_to_save["prompt"] = prompt
        response, result = zero_shot_predictor.predict_judgment(
            fact=data.fact, defendants=data.defendants
        )
        result_to_save["response"] = response
        result_to_save["result"] = [outcome.model_dump() for outcome in result]
        save_json(result_to_save, os.path.join(output_dir, f"result_{i}.json"))
        print(f"Processed case {i + 1}/{len(legal_data)}")


if __name__ == "__main__":
    main()
