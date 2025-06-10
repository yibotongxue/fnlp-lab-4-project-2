from abc import ABC, abstractmethod
import argparse
from typing import Any
import os

import evaluate
import numpy as np
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
        print(f"Training configuration: {self.train_config}")
        self.init_model()
        print(f"Model initialized: {self.model.__class__.__name__}")
        self.init_datasets()
        print("Datasets initialized.")
        self.init_metrics()
        print("Metrics initialized.")
        self.init_trainer()
        print("Trainer initialized.")

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
        self.train_dataset = CaseDataset(
            train_raw_cases[:-1000], self.tokenizer, is_train=True
        )
        self.test_dataset = CaseDataset(
            test_raw_cases[-1000:], self.tokenizer, is_train=True
        )
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
                eval_steps=self.train_config.get("eval_steps", 500),
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
                eval_steps=self.train_config.get("eval_steps", 500),
                save_strategy=self.train_config["save_strategy"],
                push_to_hub=False,
                logging_dir=self.train_config["logging_dir"],
                logging_steps=10,
            )

        self.trainer = LegalTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.test_dataset,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            charge_weight=self.train_config["charge_weight"],
            imprisonment_weight=self.train_config["imprisonment_weight"],
            compute_metrics=self.compute_metrics,
        )

    def init_metrics(self):
        """Initialize the metrics."""
        self.accuracy_metric = evaluate.load("accuracy")
        self.f1_metric = evaluate.load("f1")

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        charge_logits, imprisonment_logits = predictions
        charge_preds = np.argmax(charge_logits, axis=1)
        imprisonment_preds = np.argmax(imprisonment_logits, axis=1)
        charge_labels = labels["charge_id"]
        imprisonment_labels = labels["imprisonment"]
        charge_accuracy = self.accuracy_metric.compute(
            predictions=charge_preds, references=charge_labels
        )["accuracy"]
        imprisonment_accuracy = self.accuracy_metric.compute(
            predictions=imprisonment_preds, references=imprisonment_labels
        )["accuracy"]
        charge_f1 = self.f1_metric.compute(
            predictions=charge_preds, references=charge_labels, average="micro"
        )["f1"]
        imprisonment_f1 = self.f1_metric.compute(
            predictions=imprisonment_preds,
            references=imprisonment_labels,
            average="micro",
        )["f1"]
        predict_imprisonment_mean = np.mean(
            imprisonment_preds[imprisonment_labels != -1]
        )
        delta_imprisonment = np.abs(
            predict_imprisonment_mean
            - np.mean(imprisonment_labels[imprisonment_labels != -1])
        )
        return {
            "charge_accuracy": charge_accuracy,
            "imprisonment_accuracy": imprisonment_accuracy,
            "charge_f1": charge_f1,
            "imprisonment_f1": imprisonment_f1,
            "predict_imprisonment_mean": predict_imprisonment_mean,
            "delta_imprisonment": delta_imprisonment,
        }

    def train(self):
        """Train the model."""
        self.trainer.train()

    def eval(self) -> dict[str, float]:
        """Evaluate the model."""
        return self.trainer.evaluate()
