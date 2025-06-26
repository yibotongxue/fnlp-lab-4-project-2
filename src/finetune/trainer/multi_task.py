from typing import Any, override
import os

import numpy as np
import torch
from torch.nn import functional as F
from transformers.trainer import Trainer
from transformers import TrainingArguments
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from .base import BaseTrainer
from ..data import BothDataset, CaseCollator
from ..utils import load_pretrained_models
from ..model import LegalPredictionModel
from ...utils import load_json
from ...utils.imprisonment_mapper import get_imprisonment_mapper


class MultiTaskTrainer(BaseTrainer):

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
        self.charge_mapper = load_json(data_cfgs["charge_file_path"])
        self.imprisonment_mapper = get_imprisonment_mapper(
            imprisonment_mapper_config=data_cfgs["imprisonment_mapper_config"]
        )
        self.train_dataset = BothDataset(
            charge_mapper=self.charge_mapper,
            imprisonment_mapper=self.imprisonment_mapper.imprisonment2label,
            **common_train_args,
        )
        self.eval_dataset = BothDataset(
            charge_mapper=self.charge_mapper,
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
        charge_num = model_cfgs.pop("charge_num")
        imprisonment_num = model_cfgs.pop("imprisonment_num")
        base_model, self.tokenizer = load_pretrained_models(
            model_name_or_path,
            is_classification=False,
            **model_cfgs,
        )
        self.model = LegalPredictionModel(
            base_model,
            charge_num=charge_num,
            imprisonment_num=imprisonment_num,
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

        self.charge_weight = train_cfgs.pop("charge_weight")
        self.imprisonment_weight = train_cfgs.pop("imprisonment_weight")

        self.training_args = TrainingArguments(**train_cfgs)

        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
            compute_loss_func=self.compute_loss,
        )

    @override
    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        charge_logits, imprisonment_logits = predictions
        charge_preds = np.argmax(charge_logits, axis=1)
        imprisonment_preds = np.argmax(imprisonment_logits, axis=1)
        charge_labels = labels["charge_id"]
        imprisonment_labels = labels["imprisonment"]
        metrics = {}
        metrics["charge_accuracy"] = accuracy_score(
            y_true=charge_labels,
            y_pred=charge_preds,
        )
        metrics["imprisonment_accuracy"] = accuracy_score(
            y_true=imprisonment_labels,
            y_pred=imprisonment_preds,
        )
        metrics["charge_f1"] = f1_score(
            y_true=charge_labels,
            y_pred=charge_preds,
            average="weighted",
        )
        metrics["imprisonment_f1"] = f1_score(
            y_true=imprisonment_labels,
            y_pred=imprisonment_preds,
            average="weighted",
        )
        metrics["charge_precision"] = precision_score(
            y_true=charge_labels, y_pred=charge_preds, average="weighted"
        )
        metrics["imprisionment_precision"] = precision_score(
            y_true=imprisonment_labels, y_pred=imprisonment_preds, average="weighted"
        )
        metrics["charge_recall"] = recall_score(
            y_true=charge_labels,
            y_pred=charge_preds,
            average="weighted",
        )
        metrics["imprisonment_recall"] = recall_score(
            y_true=imprisonment_labels,
            y_pred=imprisonment_preds,
            average="weighted",
        )
        metrics["predict_imprisonment_mean"] = np.mean(
            imprisonment_preds[imprisonment_labels != -1]
        )
        metrics["delta_imprisonment"] = np.abs(
            np.mean(imprisonment_preds[imprisonment_labels != -1])
            - np.mean(imprisonment_labels[imprisonment_labels != -1])
        )
        return metrics

    def compute_loss(self, outputs, labels, num_items_in_batch=None):
        """
        Computes the loss for the model given the inputs.

        Args:
            model: The model to compute the loss for.
            inputs: The inputs to the model.
            return_outputs (bool): Whether to return the outputs of the model.

        Returns:
            The computed loss and optionally the outputs of the model.
        """
        charge_logits = outputs.charge_logits
        imprisonment_logits = outputs.imprisonment_logits
        charge_labels = labels["charge_id"]
        imprisonment_labels = labels["imprisonment"]
        charge_loss = self._compute_charge_loss(charge_logits, charge_labels)
        imprisonment_loss = self._compute_imprisonment_loss(
            imprisonment_logits, imprisonment_labels, charge_labels
        )
        total_loss = (
            charge_loss * self.charge_weight
            + imprisonment_loss * self.imprisonment_weight
        )
        return total_loss

    def _compute_charge_loss(self, charge_logits, charge_labels):
        """
        Computes the charge loss using cross-entropy loss.

        Args:
            charge_logits: The logits for the charge predictions.
            charge_labels: The labels for the charge predictions.

        Returns:
            The computed charge loss.
        """
        return F.cross_entropy(
            charge_logits,
            charge_labels,
            ignore_index=-1,  # Ignore the -1 label
            reduction="mean",  # Average loss over the batch
        )

    def _compute_imprisonment_loss(
        self, imprisonment_logits, imprisonment_labels, charge_labels
    ):
        """
        Computes the imprisonment loss using cross-entropy loss.
        Args:
            imprisonment_logits: The logits for the imprisonment predictions.
            imprisonment_labels: The labels for the imprisonment predictions.
            charge_labels: The labels for the charge predictions.
        Returns:
            The computed imprisonment loss.
        """

        # Filter out imprisonment labels where charge label is -1
        valid_mask = charge_labels != -1
        valid_imprisonment_logits = imprisonment_logits[valid_mask]
        valid_imprisonment_labels = imprisonment_labels[valid_mask]

        return F.cross_entropy(
            valid_imprisonment_logits,
            valid_imprisonment_labels,
            ignore_index=-1,  # Ignore the -1 label
            reduction="mean",  # Average loss over the batch
        )


if __name__ == "__main__":
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

    trainer = MultiTaskTrainer(cfgs=cfgs)
    trainer.train()
    print(trainer.eval())
