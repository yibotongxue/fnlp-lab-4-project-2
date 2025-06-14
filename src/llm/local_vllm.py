import requests
from typing import override

from .base import BaseLLM


class LocalVLLM(BaseLLM):
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        model_name: str = "Qwen/Qwen3-0.6B",
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name

    @override
    def generate(self, prompt: str, system_prompt: str = "") -> str:
        url = f"{self.base_url}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        data = {"model": self.model_name, "messages": []}
        if system_prompt:
            data["messages"].append({"role": "system", "content": system_prompt})
        data["messages"].append({"role": "user", "content": prompt})

        resp = requests.post(url, json=data, headers=headers)
        resp.raise_for_status()  # 如果请求失败，抛异常

        result = resp.json()
        # 提取回答文本
        return result["choices"][0]["message"]["content"]
