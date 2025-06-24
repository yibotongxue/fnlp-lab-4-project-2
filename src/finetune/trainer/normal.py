from typing import override, Any
import os

import numpy as np
import torch
from transformers import TrainingArguments, Trainer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from .base import BaseTrainer
from ..utils import load_pretrained_models
from ..data import (
    SingleChargeDataset,
    MultiChargeDataset,
    ImprisonmentDataset,
    CaseCollator,
)
from ...utils import load_json
from ...utils.imprisonment_mapper import get_imprisonment_mapper


class NormalTrainer(BaseTrainer):
    def __init__(self, cfgs: dict[str, Any]):
        super().__init__(cfgs)

    @override
    def init_datasets(self):
        """Initialize the datasets."""
        data_cfgs: dict[str, Any] = self.cfgs["data_cfgs"]
        common_train_args = {
            "path": data_cfgs["train_dataset_name_or_path"],
            "template": data_cfgs["train_dataset_template"],
            "name": data_cfgs.get("train_dataset_name", None),
            "size": data_cfgs.get("train_size", None),
            "split": "train",
            "data_files": data_cfgs.get("train_data_files", None),
            "optional_args": data_cfgs.get("train_dataset_optional_args", {}),
        }
        common_eval_args = {
            "path": data_cfgs["eval_dataset_name_or_path"],
            "template": data_cfgs["eval_dataset_template"],
            "name": data_cfgs.get("eval_dataset_name", None),
            "size": data_cfgs.get("eval_size", None),
            "split": "eval",
            "data_files": data_cfgs.get("eval_data_files", None),
            "optional_args": data_cfgs.get("eval_dataset_optional_args", {}),
        }
        if data_cfgs["type"] == "charge":
            self.charge_mapper = load_json(data_cfgs["charge_file_path"])
            self.train_dataset = SingleChargeDataset(
                charge_mapper=self.charge_mapper, **common_train_args
            )
            self.eval_dataset = SingleChargeDataset(
                charge_mapper=self.charge_mapper, **common_eval_args
            )
        elif data_cfgs["type"] == "multi_charge":
            self.charge_mapper = load_json(data_cfgs["charge_file_path"])
            self.train_dataset = MultiChargeDataset(
                charge_mapper=self.charge_mapper, **common_train_args
            )
            self.eval_dataset = MultiChargeDataset(
                charge_mapper=self.charge_mapper, **common_eval_args
            )
        elif data_cfgs["type"] == "imprisonment":
            self.imprisonment_mapper = get_imprisonment_mapper(
                imprisonment_mapper_config=data_cfgs["imprisonment_mapper_config"]
            )
            self.train_dataset = ImprisonmentDataset(
                imprisonment_mapper=self.imprisonment_mapper.imprisonment2label,
                **common_train_args,
            )
            self.eval_dataset = ImprisonmentDataset(
                imprisonment_mapper=self.imprisonment_mapper.imprisonment2label,
                **common_eval_args,
            )

        if data_cfgs["data_type"] == "bf16":
            data_type = torch.bfloat16
        elif data_cfgs["data_type"] == "f16":
            data_type = torch.float16
        elif data_cfgs["data_type"] == "long":
            data_type = torch.long
        elif data_cfgs["data_type"] == "float" or data_cfgs["data_type"] is None:
            data_type = torch.float
        else:
            print(
                f"Warning: unsupported data type {data_cfgs["data_type"]}, set to float as default"
            )
            data_type = torch.float
        self.data_collator = CaseCollator(
            tokenizer=self.tokenizer,
            max_length=data_cfgs["max_length"],
            data_type=data_type,
        )

    @override
    def init_model(self):
        """Initialize the model with pretrained weights."""
        model_cfgs = self.cfgs["model_cfgs"]
        model_name_or_path = model_cfgs.pop("model_name_or_path")
        self.model, self.tokenizer = load_pretrained_models(
            model_name_or_path,
            is_classification=True,
            **model_cfgs,
        )

    @override
    def init_trainer(self):
        """Initialize the training arguments."""
        train_cfgs: dict[str, Any] = self.cfgs["train_cfgs"]
        if ("wandb" == train_cfgs["report_to"]) or (
            isinstance(train_cfgs["report_to"], list)
            and ("wandb" in train_cfgs["report_to"])
        ):
            os.environ["WANDB_PROJECT"] = train_cfgs.pop(
                "project_name", "default_project"
            )

        self.training_args = TrainingArguments(**train_cfgs)

        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
        )

    def compute_metrics(self, eval_pred):
        """
        Compute metrics based on the evaluation predictions.

        Args:
            eval_pred: Evaluation predictions from the Trainer.

        Returns:
            dict: Computed metrics.
        """
        predictions, labels = eval_pred
        metrics = {}  # Initialize an empty dictionary to store all metrics

        if self.cfgs["data_cfgs"]["type"] == "multi_charge":
            predictions = 1 / (1 + np.exp(-predictions))
            predictions = (predictions > 0.5).astype(int)
            labels = labels.astype(int)
            metrics["accuracy"] = accuracy_score(
                y_pred=predictions,
                y_true=labels,
            )
            metrics["f1"] = f1_score(
                y_pred=predictions, y_true=labels, average="samples"
            )
            metrics["precision"] = precision_score(
                y_pred=predictions, y_true=labels, average="samples"
            )
            metrics["recall"] = recall_score(
                y_pred=predictions, y_true=labels, average="samples"
            )
        else:
            predictions = predictions.argmax(axis=-1).astype(int)
            labels = labels.astype(int).reshape(-1)
            metrics["accuracy"] = accuracy_score(
                y_pred=predictions,
                y_true=labels,
            )
            metrics["f1"] = f1_score(
                y_pred=predictions, y_true=labels, average="weighted"
            )
            metrics["precision"] = precision_score(
                y_pred=predictions, y_true=labels, average="weighted"
            )
            metrics["recall"] = recall_score(
                y_pred=predictions, y_true=labels, average="weighted"
            )

        return metrics


def main():
    import argparse

    from ...utils import load_config
    from ...utils.tools import update_dict, custom_cfgs_to_dict

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config-file-path",
        type=str,
        required=True,
        help="The path to the config file",
    )
    args, unparsed_args = parser.parse_known_args()

    cfgs = load_config(args.config_file_path)

    keys = [k[2:] for k in unparsed_args[::2]]
    values = list(unparsed_args[1::2])
    unparsed_args = dict(zip(keys, values))

    for k, v in unparsed_args.items():
        cfgs = update_dict(cfgs, custom_cfgs_to_dict(k, v))

    trainer = NormalTrainer(cfgs=cfgs)
    trainer.train()
    print(trainer.eval())


if __name__ == "__main__":
    main()
