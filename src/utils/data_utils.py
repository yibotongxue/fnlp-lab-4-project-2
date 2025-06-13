import os
from functools import cached_property

from torch.utils.data import Dataset

from .json_util import load_json, load_jsonl
from .type_utils import CaseDataDict


class ArticleLoader:
    def __init__(self, file_path: str):
        assert file_path.endswith(".json"), "File must be a JSON file"
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        self.file_path = file_path

    @cached_property
    def all_articles(self) -> dict[str, str]:
        return self._load_all_articles()

    def load_article(self, article_id: str) -> str:
        if article_id not in self.all_articles:
            raise KeyError(f"Article ID {article_id} not found in the file.")
        return self.all_articles[article_id]

    def get_article_id(self, article_content: str) -> str:
        for article_id, content in self.all_articles.items():
            if content == article_content:
                return article_id
        raise ValueError("Article content not found in the file.")

    def _load_all_articles(self) -> dict[str, str]:
        result = load_json(self.file_path)
        if not isinstance(result, dict):
            raise ValueError(
                "The JSON file must contain a dictionary mapping article IDs to content."
            )
        for key, value in result.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise ValueError(
                    "Each key must be an integer (article ID) and each value must be a string (article content)."
                )
        return result


class ChargeLoader:
    def __init__(self, file_path: str):
        assert file_path.endswith(".json"), "File must be a JSON file"
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        self.file_path = file_path

    @cached_property
    def charge_num(self) -> int:
        return len(self.all_charges.keys())

    @cached_property
    def all_charges(self) -> dict[str, int]:
        return self._load_all_charges()

    @cached_property
    def reverse_charges(self) -> dict[int, str]:
        return {v: k for k, v in self.all_charges.items()}

    def load_charge_id(self, charge_name: str) -> int:
        if charge_name not in self.all_charges:
            raise KeyError(f"Charge name '{charge_name}' not found in the file.")
        return self.all_charges[charge_name]

    def get_charge_name(self, charge_id: int) -> str:
        for name, id_ in self.all_charges.items():
            if id_ == charge_id:
                return name
        raise ValueError(f"Charge ID {charge_id} not found in the file.")

    def _load_all_charges(self) -> dict[str, int]:
        result = load_json(self.file_path)
        if not isinstance(result, dict):
            raise ValueError(
                "The JSON file must contain a dictionary mapping charge names to their IDs."
            )
        for key, value in result.items():
            if not isinstance(key, str) or not isinstance(value, int):
                raise ValueError(
                    "Each key must be a string (charge name) and each value must be an integer (charge ID)."
                )
        return result


class LegalCaseDataset(Dataset):
    def __init__(self, file_path: str):
        assert file_path.endswith(".jsonl"), "File must be a JSONL file"
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        self.file_path = file_path

    @cached_property
    def all_cases(self) -> list[CaseDataDict]:
        return self._load_all_cases()

    def __getitem__(self, index: int | slice) -> CaseDataDict:
        if isinstance(index, slice):
            return [
                self.all_cases[i] for i in range(*index.indices(len(self.all_cases)))
            ]
        elif isinstance(index, int):
            return self._get_case_by_index(index)
        else:
            raise TypeError("Index must be an integer or a slice.")

    def _get_case_by_index(self, index: int) -> CaseDataDict:
        if index < 0 or index >= len(self.all_cases):
            raise IndexError("Index out of range.")
        return self.all_cases[index]

    def __len__(self) -> int:
        return len(self.all_cases)

    def _load_all_cases(self) -> list[CaseDataDict]:
        result = load_jsonl(self.file_path)
        result = [CaseDataDict(**case) for case in result]
        return result
