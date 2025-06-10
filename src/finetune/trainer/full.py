from typing import override

from ..utils import load_pretrained_models
from .base import BaseTrainer


class FullTrainer(BaseTrainer):
    @override
    def init_model(self):
        """Initialize the model with pretrained weights."""
        self.model, self.tokenizer = load_pretrained_models(
            self.args.model_name_or_path,
            model_max_length=self.args.model_max_length,
            cache_dir=self.args.cache_dir,
            auto_model_kwargs=self.args.extra_model_kwargs,
            auto_tokenizer_kwargs=self.args.extra_tokenizer_kwargs,
        )


if __name__ == "__main__":
    import argparse
    from typing import Any
    import yaml

    def load_config(config_path: str) -> dict[str, Any]:
        """
        Load the configuration file.

        Args:
            config_path (str): Path to the configuration file.

        Returns:
            dict[str, Any]: Loaded configuration.
        """
        with open(config_path) as file:
            config = yaml.safe_load(file)
        return config

    parser = argparse.ArgumentParser(description="Full Trainer")
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/finetune/full.yaml",
        help="Path to the configuration file.",
    )
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument(
        "--train-data-path", type=str, required=True, help="Path to the training data."
    )
    parser.add_argument(
        "--test-data-path", type=str, required=True, help="Path to the test data."
    )
    parser.add_argument("--model-max-length", type=int, default=4096)
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache directory for model and tokenizer.",
    )
    parser.add_argument("--extra-model-kwargs", type=dict, default={})
    parser.add_argument("--extra-tokenizer-kwargs", type=dict, default={})
    parser.add_argument(
        "--enable-wandb", action="store_true", help="Enable Weights and Biases logging."
    )
    args = parser.parse_args()

    config = load_config(args.config)

    trainer = FullTrainer(args, train_config=config["train_config"])
    print("Model and tokenizer initialized successfully.")

    trainer.train()
    print(trainer.eval())
