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
