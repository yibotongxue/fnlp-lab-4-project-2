from abc import ABC, abstractmethod
from typing import Any, Callable, TypedDict

import torch
import transformers
from torch.utils.data import Dataset
from datasets import load_dataset
from tqdm import tqdm
from pydantic import BaseModel

from .data_formatter import (
    BaseFormatter,
    ChargeFormatteredSample,
    MultiChargeFormatteredSample,
    ImprisonmentFormatteredSample,
    BothFormatteredSample,
)
from ...utils.tools import enable_bracket_access
from .template_registry import TEMPLATE_REGISTRY


__all__ = [
    "CaseSample",
    "CaseBatch",
    "SingleChargeDataset",
    "MultiChargeDataset",
    "ImprisonmentDataset",
    "BothDataset",
    "CaseCollator",
]


@enable_bracket_access
class CaseSample(BaseModel):
    fact: str
    label: int | list[int] | dict[str, Any]


class CaseBatch(TypedDict):
    input_ids: torch.LongTensor
    labels: torch.LongTensor | dict[str, torch.LongTensor]
    attention_mask: torch.BoolTensor


class CaseDataset(Dataset, ABC):

    def __init__(
        self,
        path: str,
        template: BaseFormatter,
        name: str | None = None,
        size: int | None = None,
        split: str | None = None,
        data_files: str | None = None,
        optional_args: list | str = [],
    ):
        super().__init__()
        assert path, f"You must set the valid datasets path! Here is {path}"
        assert template, f"You must set the valid template path! Here is {template}"
        self.template = TEMPLATE_REGISTRY[template]()

        if isinstance(optional_args, str):
            optional_args = [optional_args]
        if path.endswith("json"):
            from ...utils import load_json

            self.raw_data = load_json(path)
        elif path.endswith("jsonl"):
            from ...utils import load_jsonl

            self.raw_data = load_jsonl(path)
        else:
            self.raw_data = load_dataset(
                path,
                name=name,
                split=split,
                data_files=data_files,
                *optional_args,
                trust_remote_code=True,
            )
        if size:
            size = min(size, len(self.raw_data))
            self.raw_data = self.raw_data.select(range(int(size)))
        self.index_mapper = self.filter_indices()

    @abstractmethod
    def filter_indices(self) -> list[tuple[int, int]]:
        pass

    @abstractmethod
    def preprocess(
        self, raw_sample: ChargeFormatteredSample, defendant_idx: int
    ) -> CaseSample:
        pass

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Get a tokenized data sample by index."""
        sample_idx, internal_idx = self.index_mapper[index]
        raw_sample = self.raw_data[sample_idx]
        data = self.preprocess(raw_sample, internal_idx)
        return data

    def __len__(self) -> int:
        """Get the number of samples in the dataset."""
        return len(self.index_mapper)


class ChargeDataset(CaseDataset):

    def __init__(
        self,
        path: str,
        template: BaseFormatter,
        charge_mapper: dict[str, int],
        name: str | None = None,
        size: int | None = None,
        split: str | None = None,
        data_files: str | None = None,
        optional_args: list | str = [],
    ):
        super().__init__(
            path,
            template,
            name=name,
            size=size,
            split=split,
            data_files=data_files,
            optional_args=optional_args,
        )
        self.charge_mapper = charge_mapper

    def filter_indices(self) -> list[tuple[int, int]]:
        index_mapper = []
        for i, item in tqdm(
            enumerate(self.raw_data),
            total=len(self.raw_data),
            desc="Filtering valid indices",
        ):
            is_valid, charge_cnt, _ = self.template.check_validation(item)
            if not is_valid:
                continue
            index_mapper.extend([(i, charge_idx) for charge_idx in range(charge_cnt)])
        return index_mapper


class SingleChargeDataset(ChargeDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def preprocess(
        self, raw_sample: ChargeFormatteredSample, defendant_idx: int
    ) -> CaseSample:
        formatted_sample = self.template.format_charge_sample(raw_sample)[defendant_idx]
        result_dict = {}
        result_dict["fact"] = formatted_sample["fact"]
        result_dict["label"] = self.charge_mapper[formatted_sample["charge_name"]]
        return CaseSample(**result_dict)


class MultiChargeDataset(ChargeDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def preprocess(
        self, raw_sample: MultiChargeFormatteredSample, defendant_idx: int
    ) -> CaseSample:
        formatted_sample = self.template.format_multi_charge_sample(raw_sample)[
            defendant_idx
        ]
        result_dict = {}
        result_dict["fact"] = formatted_sample["fact"]
        result_dict["label"] = [0] * len(self.charge_mapper.keys())
        for charge_name in formatted_sample["charge_list"]:
            result_dict["label"][self.charge_mapper[charge_name]] = 1
        return CaseSample(**result_dict)


class ImprisonmentDataset(CaseDataset):

    def __init__(
        self,
        path: str,
        template: BaseFormatter,
        imprisonment_mapper: Callable[[int], int],
        name: str | None = None,
        size: int | None = None,
        split: str | None = None,
        data_files: str | None = None,
        optional_args: list | str = [],
    ):
        super().__init__(
            path,
            template,
            name=name,
            size=size,
            split=split,
            data_files=data_files,
            optional_args=optional_args,
        )
        self.imprisonment_mapper = imprisonment_mapper

    def filter_indices(self) -> list[tuple[int, int]]:
        index_mapper = []
        for i, item in tqdm(
            enumerate(self.raw_data),
            total=len(self.raw_data),
            desc="Filtering valid indices",
        ):
            is_valid, _, imprisonment_cnt = self.template.check_validation(item)
            if not is_valid:
                continue
            index_mapper.extend(
                [(i, charge_idx) for charge_idx in range(imprisonment_cnt)]
            )
        return index_mapper

    def preprocess(
        self, raw_sample: ImprisonmentFormatteredSample, internal_idx: int
    ) -> CaseSample:
        formatted_sample = self.template.format_imprisonment_sample(raw_sample)[
            internal_idx
        ]
        result_dict = {}
        result_dict["fact"] = formatted_sample["fact"]
        result_dict["label"] = self.imprisonment_mapper(
            formatted_sample["imprisonment"]
        )
        return CaseSample(**result_dict)


class BothDataset(CaseDataset):
    def __init__(
        self,
        path: str,
        template: BaseFormatter,
        charge_mapper: dict[str, int],
        imprisonment_mapper: Callable[[int], int],
        name: str | None = None,
        size: int | None = None,
        split: str | None = None,
        data_files: str | None = None,
        optional_args: list | str = [],
    ):
        super().__init__(
            path,
            template,
            name=name,
            size=size,
            split=split,
            data_files=data_files,
            optional_args=optional_args,
        )
        self.charge_mapper = charge_mapper
        self.imprisonment_mapper = imprisonment_mapper

    def filter_indices(self) -> list[tuple[int, int]]:
        index_mapper = []
        for i, item in tqdm(
            enumerate(self.raw_data),
            total=len(self.raw_data),
            desc="Filtering valid indices",
        ):
            is_valid, charge_cnt, _ = self.template.check_validation(item)
            if not is_valid:
                continue
            index_mapper.extend([(i, charge_idx) for charge_idx in range(charge_cnt)])
        return index_mapper

    def preprocess(
        self, raw_sample: BothFormatteredSample, internal_idx: int
    ) -> CaseSample:
        formatted_sample = self.template.format_both_sample(raw_sample)[internal_idx]
        result_dict = {}
        result_dict["fact"] = formatted_sample["fact"]
        result_dict["label"] = {
            "charge_id": self.charge_mapper[formatted_sample["charge_name"]],
            "imprisonment": self.imprisonment_mapper(formatted_sample["imprisonment"]),
        }
        return CaseSample(**result_dict)


class CaseCollator:
    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        max_length: int = 512,
        data_type: torch.dtype = torch.float,
    ) -> None:
        """Initialize a collator."""
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data_type = data_type

    def __call__(self, samples: list[CaseSample]) -> CaseBatch:
        return_dict = {}
        concated_text = [sample["fact"] for sample in samples]
        tokenized_input = self.tokenizer(
            text=concated_text,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=False,
        )
        return_dict["input_ids"] = tokenized_input["input_ids"]
        return_dict["attention_mask"] = tokenized_input["attention_mask"]
        if isinstance(samples[0]["label"], dict):
            return_dict["labels"] = {}
            for k in samples[0]["label"].keys():
                return_dict["labels"][k] = torch.tensor(
                    [sample["label"][k] for sample in samples],
                    dtype=self.data_type,
                )
        else:
            return_dict["labels"] = torch.tensor(
                [sample["label"] for sample in samples],
                dtype=self.data_type,
            )

        return CaseBatch(**return_dict)
