from typing import Optional, override

from openai import OpenAI

from .base import BaseLLM


class _BaseOpenAILLM(BaseLLM):
    def __init__(self, api_key: str, base_url: Optional[str], model_name):
        """
        Initialize the OpenAI LLM with the provided API key and base URL.

        Args:
            api_key (str): Your OpenAI API key.
            base_url (str): The base URL for the OpenAI API.
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

    @override
    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """
        Generate a response based on the prompt using OpenAI's API.

        Args:
            prompt (str): The input prompt for the LLM.

        Returns:
            str: The generated response from the LLM.
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content


class OpenAILLM(_BaseOpenAILLM):
    def __init__(self, api_key: str, model_name: str = "gpt-4.1"):
        """
        Initialize the GPT LLM with the provided API key.
        Args:
            api_key (str): Your OpenAI API key.
        """
        super(api_key, None, model_name)


class QwenLLM(_BaseOpenAILLM):
    def __init__(self, api_key: str, model_name: str = "qwen-max"):
        """
        Initialize the QwenMax LLM with the provided API key.
        Args:
            api_key (str): Your OpenAI API key.
        """
        super(api_key, "https://dashscope.aliyuncs.com/compatible-mode/v1", model_name)


class DeepSeekLLM(_BaseOpenAILLM):
    def __init__(self, api_key: str, model_name: str = "deepseek-chat"):
        """
        Initialize the DeepSeek LLM with the provided API key.
        Args:
            api_key (str): Your OpenAI API key.
        """
        super().__init__(api_key, "https://api.deepseek.com", model_name)
