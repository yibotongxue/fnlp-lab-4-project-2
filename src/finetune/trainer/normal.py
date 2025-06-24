import argparse
from typing import override, Any
import os

from transformers import TrainingArguments, Trainer

from .base import BaseTrainer
from ..utils import load_pretrained_models
from ..data import ChargeDataset, ImprisonmentDataset, CaseCollator
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
            "path": data_cfgs.train_dataset_name_or_path,
            "template": data_cfgs.train_dataset_template,
            "tokenizer": self.tokenizer,
            "name": data_cfgs.get("train_dataset_name", None),
            "size": int(data_cfgs.get("train_size", None)),
            "split": "train",
            "data_files": data_cfgs.get("train_data_files", None),
            "optional_args": data_cfgs.get("train_dataset_optional_args", {}),
        }
        common_eval_args = {
            "path": data_cfgs.eval_dataset_name_or_path,
            "template": data_cfgs.eval_dataset_template,
            "tokenizer": self.tokenizer,
            "name": data_cfgs.get("eval_dataset_name", None),
            "size": int(data_cfgs.get("eval_size", None)),
            "split": "eval",
            "data_files": data_cfgs.get("eval_data_files", None),
            "optional_args": data_cfgs.get("eval_dataset_optional_args", {}),
        }
        if data_cfgs["type"] == "charge":
            self.charge_mapper = load_json(data_cfgs["charge_file_path"])
            self.train_dataset = ChargeDataset(
                charge_mapper=self.charge_mapper, **common_train_args
            )
            self.eval_dataset = ChargeDataset(
                charge_mapper=self.charge_mapper, **common_eval_args
            )
        elif data_cfgs["type"] == "imprisonment":
            self.imprisonment_mapper = get_imprisonment_mapper(
                imprisonment_mapper_config=data_cfgs["imprisonment_mapper_config"]
            )
            self.train_dataset = ImprisonmentDataset(
                imprisonment_mapper=self.imprisonment_mapper, **common_train_args
            )
            self.eval_dataset = ImprisonmentDataset(
                imprisonment_mapper=imprisonment_trainer, **common_eval_args
            )

        self.data_collator = CaseCollator(
            tokenizer=self.tokenizer,
        )

    @override
    def init_model(self):
        """Initialize the model with pretrained weights."""
        self.model, self.tokenizer = load_pretrained_models(
            self.args.model_name_or_path,
            model_max_length=self.args.model_max_length,
            cache_dir=self.args.cache_dir,
            is_classification=True,
            auto_model_kwargs=self.args.extra_model_kwargs,
            auto_tokenizer_kwargs=self.args.extra_tokenizer_kwargs,
        )

    @override
    def init_trainer(self):
        """Initialize the training arguments."""
        train_cfgs: dict[str, Any] = self.cfgs["train_cfgs"]
        training_args = {
            "output_dir": train_cfgs.get("output_dir", "trainer_output"),
            "learning_rate": float(self.train_config["learning_rate"]),
            "per_device_train_batch_size": int(self.train_config["train_batch_size"]),
            "per_device_eval_batch_size": int(self.train_config["eval_batch_size"]),
            "num_train_epochs": int(self.train_config["epochs"]),
            "weight_decay": float(self.train_config["weight_decay"]),
            "eval_strategy": self.train_config["eval_strategy"],
            "eval_steps": self.train_config.get("eval_steps", 500),
            "save_strategy": self.train_config["save_strategy"],
            "push_to_hub": False,
            "report_to": "none",
            "logging_steps": 10,
            "fp16": True,
            "dataloader_num_workers": 4,
            "gradient_accumulation_steps": 2,
        }
        if self.args.enable_wandb:
            os.environ["WANDB_PROJECT"] = self.train_config.get(
                "wandb_project", "default_project"
            )
            training_args["report_to"] = "wandb"
            training_args["run_name"] = self.train_config["run_name"]

        self.training_args = TrainingArguments(**training_args)

        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.test_dataset,
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
        predictions = predictions.argmax(axis=-1)
        accuracy = self.accuracy_metric.compute(
            predictions=predictions, references=labels
        )
        f1 = self.f1_metric.compute(
            predictions=predictions, references=labels, average="weighted"
        )
        return {"accuracy": accuracy["accuracy"], "f1": f1["f1"]}


if __name__ == "__main__":
    from typing import Any

    from ...utils import load_config

    def parse_dict_arg(arg_values):
        kwargs = {}
        for arg in arg_values:
            if "=" in arg:
                key, value = arg.split("=", 1)
                try:
                    # Attempt to convert to float or int if possible
                    value = float(value) if "." in value else int(value)
                except ValueError:
                    pass  # Keep as string if conversion fails
                kwargs[key] = value
            else:
                print(f"Warning: Ignoring malformed key-value pair: {arg}")
        return kwargs

    parser = argparse.ArgumentParser(description="Normal Trainer")
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/finetune/normal.yaml",
        help="Path to the configuration file.",
    )
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument(
        "--train-data-path", type=str, required=True, help="Path to the training data."
    )
    parser.add_argument(
        "--test-data-path", type=str, required=True, help="Path to the test data."
    )
    parser.add_argument("--model-max-length", type=int, default=512)
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache directory for model and tokenizer.",
    )
    parser.add_argument("--extra-model-kwargs", nargs="+", type=str, default=None)
    parser.add_argument("--extra-tokenizer-kwargs", nargs="+", type=str, default=None)
    parser.add_argument(
        "--enable-wandb", action="store_true", help="Enable Weights and Biases logging."
    )
    parser.add_argument(
        "--is-charge",
        action="store_true",
        help="Whether to train the charge prediction model (default: True, for imprisonment prediction if False).",
    )
    args = parser.parse_args()

    if args.extra_model_kwargs:
        args.extra_model_kwargs = parse_dict_arg(args.extra_model_kwargs)
    if args.extra_tokenizer_kwargs:
        args.extra_tokenizer_kwargs = parse_dict_arg(args.extra_tokenizer_kwargs)

    config = load_config(args.config)

    if args.is_charge:
        print("Training charge prediction model.")
        charge_trainer = NormalTrainer(
            args, train_config=config["train_config"], is_charge=True
        )
        charge_trainer.train()
        print(charge_trainer.eval())
    else:
        print("Training imprisonment prediction model.")
        imprisonment_trainer = NormalTrainer(
            args, train_config=config["train_config"], is_charge=False
        )
        imprisonment_trainer.train()
        print(imprisonment_trainer.eval())
