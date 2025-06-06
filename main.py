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
    output_dir = "./output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(os.path.join(output_dir, "zero_shot")):
        os.makedirs(os.path.join(output_dir, "zero_shot"))
    legal_data = get_data("train")[2:4]
    print(f"Loaded {len(legal_data)} cases from the dataset.")
    llm = get_llm(
        "qwen", api_key=os.getenv("QWEN_API_KEY"), model_name="qwen-max-latest"
    )
    zero_shot_predictor = ZeroShotPredictor(llm)
    for i, data in enumerate(legal_data):
        result_to_save = {}
        result_to_save["input"] = data.model_dump()
        prompt = zero_shot_predictor.build_zero_shot_prompt(
            fact=data.fact, defendants=data.defendants
        )
        result_to_save["prompt"] = prompt
        response, result = zero_shot_predictor.predict_judgment(
            fact=data.fact, defendants=data.defendants
        )
        result_to_save["response"] = response
        result_to_save["result"] = [outcome.model_dump() for outcome in result]
        save_json(
            result_to_save, os.path.join(output_dir, "zero_shot", f"result_{i}.json")
        )
        print(f"Processed case {i + 1}/{len(legal_data)}")


if __name__ == "__main__":
    main()


import sys

sys.exit(0)

load_dotenv()

train_data = LegalCaseDataSet("./data/train.jsonl")
train_data = train_data[2]
print(train_data)

deepseek = QwenLLM(api_key=os.getenv("QWEN_API_KEY"))

zero_shot_predictor = ZeroShotPredictor(deepseek)

prompt = zero_shot_predictor.build_zero_shot_prompt(
    fact=train_data.fact, defendants=train_data.defendants
)

# print(prompt)

result = zero_shot_predictor.predict_judgment(
    fact=train_data.fact, defendants=train_data.defendants
)

print(result)
