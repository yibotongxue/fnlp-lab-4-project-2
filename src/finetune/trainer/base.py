from abc import ABC, abstractmethod
import argparse
from typing import Any
import os

from transformers import TrainingArguments

from ..data import CaseDataset, CustomDataCollator
from .hf_trainer import LegalTrainer
from ...utils import load_jsonl


class BaseTrainer(ABC):
    def __init__(
        self, args: argparse.Namespace, train_config: dict[str, Any] | None = None
    ):
        self.args = args
        self.train_config = train_config or {}
        self.init_model()
        self.init_datasets()
        self.init_trainer()

    @abstractmethod
    def init_model(self):
        """Initialize the model."""
        raise NotImplementedError("Subclasses should implement this method.")

    def init_datasets(self):
        """Initialize the datasets."""
        train_data_path = self.args.train_data_path
        test_data_path = self.args.test_data_path
        train_raw_cases = load_jsonl(train_data_path)
        test_raw_cases = load_jsonl(test_data_path)
        self.train_dataset = CaseDataset(train_raw_cases, self.tokenizer, is_train=True)
        self.test_dataset = CaseDataset(test_raw_cases, self.tokenizer, is_train=False)
        self.data_collator = CustomDataCollator()

    def init_trainer(self):
        """Initialize the training arguments."""
        if self.args.enable_wandb:
            os.environ["WANDB_PROJECT"] = self.train_config.get(
                "wandb_project", "default_project"
            )
            self.training_args = TrainingArguments(
                output_dir=self.train_config["output_dir"],
                learning_rate=float(self.train_config["learning_rate"]),
                per_device_train_batch_size=int(self.train_config["train_batch_size"]),
                per_device_eval_batch_size=int(self.train_config["eval_batch_size"]),
                num_train_epochs=int(self.train_config["epochs"]),
                weight_decay=float(self.train_config["weight_decay"]),
                eval_strategy=self.train_config["eval_strategy"],
                save_strategy=self.train_config["save_strategy"],
                push_to_hub=False,
                report_to="wandb",
                run_name=self.train_config["run_name"],
                logging_dir=self.train_config["logging_dir"],
                logging_steps=10,
            )
        else:
            self.training_args = TrainingArguments(
                output_dir=self.train_config["output_dir"],
                learning_rate=float(self.train_config["learning_rate"]),
                per_device_train_batch_size=int(self.train_config["train_batch_size"]),
                per_device_eval_batch_size=int(self.train_config["eval_batch_size"]),
                num_train_epochs=int(self.train_config["epochs"]),
                weight_decay=float(self.train_config["weight_decay"]),
                eval_strategy=self.train_config["eval_strategy"],
                save_strategy=self.train_config["save_strategy"],
                push_to_hub=False,
                logging_dir=self.train_config["logging_dir"],
                logging_steps=10,
            )

        self.trainer = LegalTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.tokenized_datasets["train"],
            eval_dataset=self.tokenized_datasets["test"],
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            charge_weight=self.train_config["charge_weight"],
            imprisonment_weight=self.train_config["imprisonment_weight"],
        )

    def train(self):
        """Train the model."""
        self.trainer.train()

    def eval(self) -> dict[str, float]:
        """Evaluate the model."""
        return self.trainer.evaluate()
