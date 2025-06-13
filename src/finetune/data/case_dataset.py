import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class CaseDataset(Dataset):
    def __init__(
        self,
        raw_cases: list[dict],
        tokenizer: PreTrainedTokenizer,
        is_train: bool = True,
        with_accusation: bool = False,
    ):
        self.raw_cases = raw_cases
        self.tokenizer = tokenizer
        self.is_train = (is_train,)
        self.with_accusation = with_accusation

    def __len__(self):
        return len(self.raw_cases)

    def __getitem__(self, idx: int | slice) -> dict:
        cases = self.raw_cases[idx]
        if isinstance(idx, int):
            return self._prepare_case(cases)
        elif isinstance(idx, slice):
            return [self._prepare_case(case) for case in cases]
        else:
            raise TypeError("Index must be an integer or a slice.")

    def _prepare_case(self, case: dict) -> dict:
        if self.with_accusation:
            text = case["fact_imprisonment"]
        else:
            text = case["fact"]
        # fact = case["fact"]

        encoding = self.tokenizer(
            text=text,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        if not self.is_train:
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }

        charge_id = case.get("charge_id", -1)
        imprisonment = self._imprisonment_to_class(case.get("imprisonment", -1))

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": {
                "charge_id": torch.tensor(charge_id, dtype=torch.long),
                "imprisonment": torch.tensor(imprisonment, dtype=torch.long),
            },
        }

    @staticmethod
    def _imprisonment_to_class(imprisonment: int) -> int:
        if imprisonment <= 47:
            return imprisonment
        return 48 + (imprisonment - 48) // 6

    @staticmethod
    def _class_to_imprisonment(class_id: int) -> int:
        if class_id <= 47:
            return class_id
        return 48 + (class_id - 48) * 6

    @staticmethod
    def imprisonment_class_cnt() -> int:
        # total up of imprisonment is 180
        # turn to class is
        return 74
