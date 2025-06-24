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
                imprisonment_mapper=self.imprisonment_mapper.imprisonment2label,
                **common_train_args,
            )
            self.eval_dataset = ImprisonmentDataset(
                imprisonment_mapper=self.imprisonment_mapper.imprisonment2label,
                **common_eval_args,
            )

        self.data_collator = CaseCollator(
            tokenizer=self.tokenizer,
            max_length=data_cfgs["max_length"],
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
        predictions = predictions.argmax(axis=-1)
        accuracy = self.accuracy_metric.compute(
            predictions=predictions, references=labels
        )
        f1 = self.f1_metric.compute(
            predictions=predictions, references=labels, average="micro"
        )
        return {"accuracy": accuracy["accuracy"], "f1": f1["f1"]}


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
