import argparse

from src.predictor import ZeroShotPredictor
from src.utils import save_jsonl
from src.utils.data_utils import LegalCaseDataset


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
        "--output-file",
        type=str,
        default="batch.jsonl",
        help="File to save the zero-shot prediction requests in JSONL format",
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
    split = args.split
    model_name = args.model_name
    start_index = args.start_index
    train_size = args.train_size

    legal_data = get_data(split)
    if train_size is not None:
        legal_data = legal_data[start_index : start_index + train_size]
    else:
        legal_data = legal_data[start_index:]
    print(f"Loaded {len(legal_data)} cases from the dataset.")
    jsonl_data = []
    url_dict = {
        "batch-test-model": "/v1/chat/ds-test",
    }
    for i, data in enumerate(legal_data):
        result_to_save = {}
        result_to_save["custom_id"] = str(i + 1)
        result_to_save["method"] = "POST"
        result_to_save["url"] = url_dict.get(model_name, "/v1/chat/completions")
        result_to_save["body"] = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": ZeroShotPredictor.system_prompt},
                {
                    "role": "user",
                    "content": ZeroShotPredictor.build_zero_shot_prompt(
                        fact=data.fact, defendants=data.defendants
                    ),
                },
            ],
        }
        jsonl_data.append(result_to_save)
        print(f"Processed case {i + 1}/{len(legal_data)}")
    save_jsonl(jsonl_data, args.output_file)
    print(f"Zero-shot prediction requests saved to {args.output_file}")


if __name__ == "__main__":
    main()
