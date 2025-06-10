import argparse
import os

from dotenv import load_dotenv

from src.predictor import ZeroShotPredictor
from src.utils import load_jsonl, robust_dict_from_str, save_json
from src.utils.data_utils import LegalCaseDataset

load_dotenv()


def get_data(split: str = "train") -> LegalCaseDataset:
    """Load the legal case dataset."""
    if split == "train":
        return LegalCaseDataset("./data/train.jsonl")
    elif split == "test":
        return LegalCaseDataset("./data/test.jsonl")
    else:
        raise ValueError("Invalid split. Use 'train' or 'test'.")


def main():
    parser = argparse.ArgumentParser(description="Zero-shot legal case prediction")
    parser.add_argument(
        "--input-file",
        type=str,
        default="./result.jsonl",
        help="File containing the zero-shot prediction results in JSONL format",
    )
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
    input_file = args.input_file
    output_dir = args.output_dir
    split = args.split
    start_index = args.start_index
    train_size = args.train_size

    jsonl_data = load_jsonl(input_file)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    legal_data = get_data(split)
    if train_size is not None:
        legal_data = legal_data[start_index : start_index + train_size]
    else:
        legal_data = legal_data[start_index:]
    print(f"Loaded {len(legal_data)} cases from the dataset.")
    for i, data in enumerate(legal_data):
        result_to_save = {}
        result_to_save["input"] = data.model_dump()
        prompt = ZeroShotPredictor.build_zero_shot_prompt(
            fact=data.fact, defendants=data.defendants
        )
        result_to_save["system_prompt"] = ZeroShotPredictor.system_prompt
        result_to_save["prompt"] = prompt
        response = jsonl_data[i]["response"]["body"]["choices"][0]["message"]["content"]
        result = robust_dict_from_str(response)["outcomes"]
        result_to_save["response"] = response
        result_to_save["result"] = [outcome for outcome in result]
        save_json(result_to_save, os.path.join(output_dir, f"{i + start_index}.json"))
        print(f"Processed case {i + 1}/{len(legal_data)}")


if __name__ == "__main__":
    main()
