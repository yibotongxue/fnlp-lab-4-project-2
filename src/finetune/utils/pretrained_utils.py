import os
import warnings
from typing import Any, Callable

import torch.nn as nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from .constants import (
    DEFAULT_BOS_TOKEN,
    DEFAULT_EOS_TOKEN,
    DEFAULT_PAD_TOKEN,
    DEFAULT_UNK_TOKEN,
)


def resize_tokenizer_embedding(
    tokenizer: PreTrainedTokenizerBase, model: PreTrainedModel
) -> None:
    """Resize tokenizer and embedding."""

    def _verify_vocabulary_embedding_sizes(
        tokenizer: PreTrainedTokenizerBase,
        model: PreTrainedModel,
        format_message: Callable[[Any, Any], str],
    ) -> None:
        input_embeddings = model.get_input_embeddings()
        output_embeddings = model.get_output_embeddings()
        if input_embeddings is not None and input_embeddings.num_embeddings != len(
            tokenizer
        ):
            warnings.warn(
                format_message(len(tokenizer), input_embeddings.num_embeddings),
                category=RuntimeWarning,
                stacklevel=3,
            )
        if output_embeddings is not None and output_embeddings.num_embeddings != len(
            tokenizer
        ):
            warnings.warn(
                format_message(len(tokenizer), output_embeddings.num_embeddings),
                category=RuntimeWarning,
                stacklevel=3,
            )

    def _init_new_embeddings(
        embeddings: nn.Embedding | nn.Linear | None,
        new_num_embeddings: int,
        num_new_embeddings: int,
    ) -> None:
        if embeddings is None:
            return

        params = [embeddings.weight, getattr(embeddings, "bias", None)]
        for param in params:
            if param is None:
                continue
            assert param.size(0) == new_num_embeddings
            param_data = param.data
            param_mean = param_data[:-num_new_embeddings].mean(dim=0, keepdim=True)
            param_data[-num_new_embeddings:] = param_mean

    _verify_vocabulary_embedding_sizes(
        tokenizer=tokenizer,
        model=model,
        format_message=(
            "The tokenizer vocabulary size ({}) is different from "
            "the model embedding size ({}) before resizing."
        ).format,
    )

    special_tokens_dict = {}
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    new_num_embeddings = len(tokenizer)

    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    if num_new_tokens > 0:
        model.resize_token_embeddings(new_num_embeddings)
        _init_new_embeddings(
            model.get_input_embeddings(),
            new_num_embeddings=new_num_embeddings,
            num_new_embeddings=num_new_tokens,
        )
        _init_new_embeddings(
            model.get_output_embeddings(),
            new_num_embeddings=new_num_embeddings,
            num_new_embeddings=num_new_tokens,
        )

    _verify_vocabulary_embedding_sizes(
        tokenizer=tokenizer,
        model=model,
        format_message=(
            "The tokenizer vocabulary size ({}) is different from "
            "the model embedding size ({}) after resizing."
        ).format,
    )


def load_pretrained_models(
    model_name_or_path: str | os.PathLike,
    /,
    model_max_length: int = 512,
    cache_dir: str | os.PathLike | None = None,
    *,
    auto_model_args: tuple[Any, ...] = (),
    auto_model_kwargs: dict[str, Any] | None = None,
    auto_tokenizer_args: tuple[Any, ...] = (),
    auto_tokenizer_kwargs: dict[str, Any] | None = None,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load pretrained models and tokenizer."""
    model_name_or_path = os.path.expanduser(model_name_or_path)
    cache_dir = os.path.expanduser(cache_dir) if cache_dir is not None else None
    if auto_model_kwargs is None:
        auto_model_kwargs = {}
    if auto_tokenizer_kwargs is None:
        auto_tokenizer_kwargs = {}

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        *auto_model_args,
        cache_dir=cache_dir,
        **auto_model_kwargs,
    )
    print(auto_model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        *auto_tokenizer_args,
        cache_dir=cache_dir,
        model_max_length=model_max_length,
        **auto_tokenizer_kwargs,
    )
    resize_tokenizer_embedding(tokenizer=tokenizer, model=model)
    return model, tokenizer
