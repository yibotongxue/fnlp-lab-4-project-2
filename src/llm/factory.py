from .base import BaseLLM
from .openai import DeepSeekLLM, OpenAILLM, QwenLLM


def get_llm(model: str, api_key: str = None, model_name: str = None) -> BaseLLM:
    """
    Factory function to get an instance of a language model based on the model name.

    Args:
        model_name (str): The name of the model to instantiate.
        api_key (str, optional): API key for the model if required.

    Returns:
        BaseLLM: An instance of the specified language model.
    """
    if model == "openai":
        if not model_name:
            return OpenAILLM(api_key=api_key)
        return OpenAILLM(api_key=api_key, model_name=model_name)
    elif model == "qwen":
        if not model_name:
            return QwenLLM(api_key=api_key)
        return QwenLLM(api_key=api_key, model_name=model_name)
    elif model == "deepseek":
        if not model_name:
            return DeepSeekLLM(api_key=api_key)
        return DeepSeekLLM(api_key=api_key, model_name=model_name)
    else:
        raise ValueError(f"Unsupported model name: {model}")
