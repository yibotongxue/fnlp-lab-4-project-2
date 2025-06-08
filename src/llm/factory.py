import os

from .base import BaseLLM
from .openai import DeepSeekLLM, OpenAILLM, QwenLLM


def get_llm(model_name: str = "qwen-max", api_key: str = None) -> BaseLLM:
    """
    Factory function to get an instance of a language model based on the model name.

    Args:
        model_name (str): The name of the model to instantiate.
        api_key (str, optional): API key for the model if required.

    Returns:
        BaseLLM: An instance of the specified language model.
    """
    openai_model_names = ["gpt-4.1"]
    deepseek_model_names = ["deepseek-chat", "deepseek-reasoner"]
    qwen_model_names = [
        "qwen-max",
        "qwen-plus",
        "qwen-max-latest",
    ]
    if api_key is None:
        if model_name in openai_model_names:
            api_key = os.getenv("OPENAI_API_KEY")
        elif model_name in deepseek_model_names:
            api_key = os.getenv("DEEPSEEK_API_KEY")
        elif model_name in qwen_model_names:
            api_key = os.getenv("QWEN_API_KEY")
    if model_name in openai_model_names:
        return OpenAILLM(api_key=api_key, model_name=model_name)
    elif model_name in deepseek_model_names:
        return DeepSeekLLM(api_key=api_key, model_name=model_name)
    elif model_name in qwen_model_names:
        return QwenLLM(api_key=api_key, model_name=model_name)
    else:
        raise ValueError(
            f"Unsupported model name: {model_name}. Supported models are: {openai_model_names + deepseek_model_names + qwen_model_names}."
        )
