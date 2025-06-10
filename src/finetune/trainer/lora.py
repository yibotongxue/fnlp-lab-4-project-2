from typing import override

from peft import get_peft_model, LoraConfig, TaskType

from .base import BaseTrainer
from ..utils import load_pretrained_models


class LoRATrainer(BaseTrainer):
    @override
    def init_model(self):
        function_model, self.tokenizer = load_pretrained_models(
            self.args.model_name_or_path,
            model_max_length=self.args.model_max_length,
            cache_dir=self.args.cache_dir,
            auto_model_kwargs=self.args.extra_model_kwargs,
            auto_tokenizer_kwargs=self.args.extra_tokenizer_kwargs,
        )
        self.lora_config = LoraConfig(
            r=self.train_config.lora_config.rank,
            lora_alpha=self.train_config.lora_config.alpha,
            target_modules=["query", "value"],
            lora_dropout=self.train_config.lora_config.dropout,
            bias="lora_only",
            task_type=TaskType.SEQ_CLS,
        )
        self.model = get_peft_model(function_model, self.lora_config)
