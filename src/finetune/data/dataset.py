from abc import ABC, abstractmethod
from typing import Callable

import torch
import transformers
from torch.utils.data import Dataset
from datasets import load_dataset
from tqdm import tqdm
from transformers.tokenization_utils import PaddingStrategy, TruncationStrategy
from pydantic import BaseModel

from .data_formatter import (
    BaseFormatter,
    ChargeFormatteredSample,
)
from ...utils.type_utils import enable_bracket_access


@enable_bracket_access
class CaseSample(BaseModel):
    fact: str
    label: int


@enable_bracket_access
class CaseBatch(BaseModel):
    input_ids: torch.LongTensor
    labels: torch.LongTensor
    attention_mask: torch.BoolTensor


class CaseDataset(Dataset, ABC):

    def __init__(
        self,
        path: str,
        template: BaseFormatter,
        tokenizer: transformers.PreTrainedTokenizer,
        name: str | None = None,
        size: int | None = None,
        split: str | None = None,
        data_files: str | None = None,
        optional_args: list | str = [],
    ):
        super().__init__()
        assert path, f"You must set the valid datasets path! Here is {path}"
        assert template, f"You must set the valid template path! Here is {template}"
        self.tokenizer = tokenizer
        self.template = template

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

    def get_collator(
        self, device: torch.device
    ) -> Callable[[list[CaseSample]], CaseBatch]:
        return CaseCollator(tokenizer=self.tokenizer, device=device)

    def tokenize(
        self,
        fact: str,
        add_special_tokens: bool = False,
        padding: bool | str | PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation: bool | str | TruncationStrategy = TruncationStrategy.LONGEST_FIRST,
        max_length: int | None = None,
    ) -> torch.LongTensor:  # size = (L,)
        """Tokenize a text string into a tensor representation."""
        if max_length is None:
            max_length = self.tokenizer.model_max_length

        return self.tokenizer(
            text=fact,
            add_special_tokens=add_special_tokens,
            padding=padding,
            max_length=max_length,
            truncation=truncation,
            return_tensors="pt",
        )

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
        tokenizer: transformers.PreTrainedTokenizer,
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
            tokenizer,
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

    def preprocess(
        self, raw_sample: ChargeFormatteredSample, defendant_idx: int
    ) -> CaseSample:
        formatted_sample = self.template.format_charge_sample(raw_sample)[defendant_idx]
        result_dict = {}
        result_dict["fact"] = formatted_sample["fact"]
        result_dict["label"] = self.charge_mapper[formatted_sample["charge_name"]]
        return CaseSample(**result_dict)


class ImprisonmentDataset(CaseDataset):

    def __init__(
        self,
        path: str,
        template: BaseFormatter,
        tokenizer: transformers.PreTrainedTokenizer,
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
            tokenizer,
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
        self, raw_sample: ChargeFormatteredSample, defendant_idx: int
    ) -> CaseSample:
        formatted_sample = self.template.format_imprisonment_sample(raw_sample)[
            defendant_idx
        ]
        result_dict = {}
        result_dict["fact"] = formatted_sample["fact"]
        result_dict["label"] = self.imprisonment_mapper(
            formatted_sample["imprisonment"]
        )
        return CaseSample(**result_dict)


class CaseCollator:
    def __init__(
        self, tokenizer: transformers.PreTrainedTokenizer, device: torch.device
    ) -> None:
        """Initialize a collator."""
        self.tokenizer = tokenizer
        self.device = device

    def __call__(self, samples: list[CaseSample]) -> CaseBatch:
        return_dict = {}
        concated_text = [sample["fact"] for sample in samples]
        tokenized_input = self.tokenizer(
            text=concated_text,
            return_tensors="pt",
            padding=True,
            padding_side=self.padding_side,
            return_attention_mask=True,
            add_special_tokens=False,
        )
        return_dict["input_ids"] = tokenized_input["input_ids"].to(self.device)
        return_dict["attention_mask"] = tokenized_input["attention_mask"].to(
            self.device
        )
        return_dict["labels"] = torch.tensor(
            [sample["label"] for sample in samples],
            dtype=torch.long,
            device=self.device,
        )

        return CaseBatch(**return_dict)
