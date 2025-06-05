from abc import ABC, abstractmethod


class BaseLLM(ABC):
    @abstractmethod
    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """
        Generate a response based on the prompt.

        Args:
            prompt (str): The input prompt for the LLM.

        Returns:
            str: The generated response from the LLM.
        """

    def generate_batch(self, prompts: list[str], system_prompt: str = "") -> list[str]:
        """
        Generate responses for a batch of prompts.

        Args:
            prompts (list[str]): A list of input prompts for the LLM.

        Returns:
            list[str]: A list of generated responses from the LLM.
        """
        for prompt in prompts:
            if not isinstance(prompt, str):
                raise ValueError("All prompts must be strings.")
        return [
            self.generate(prompt, system_prompt=system_prompt) for prompt in prompts
        ]
