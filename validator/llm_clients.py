"""
LLM Clients — 多LLM客户端注册表
支持注册不同的LLM作为标注验证者
"""

import json
import os
import subprocess
from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class BaseLLMClient(ABC):
    """LLM客户端基类"""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def classify_batch(self, samples: List[Dict], label_system: str) -> List[Dict]:
        """对一批样本进行分类标注

        Args:
            samples: [{"idx": int, "title": str, "content": str, ...}]
            label_system: 标签体系说明文本

        Returns:
            [{"idx": int, "label": str, "confidence": float, "reasoning": str}]
        """
        ...


class DeepSeekClient(BaseLLMClient):
    """DeepSeek API 客户端"""

    def __init__(self, api_key: Optional[str] = None):
        super().__init__("DeepSeek V4 Flash")
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        self.model = "deepseek-v4-flash"

    def classify_batch(self, samples, label_system):
        # 子代理方案：通过 Hermes delegate_task 调用
        # 实际运行时由 engine.py 处理
        raise NotImplementedError("Use delegate_task based dispatch")


class OpenAICompatibleClient(BaseLLMClient):
    """兼容 OpenAI API 格式的 LLM"""

    def __init__(self, name: str, api_key: str, base_url: str, model: str):
        super().__init__(name)
        self.api_key = api_key
        self.base_url = base_url
        self.model = model

    def classify_batch(self, samples, label_system):
        raise NotImplementedError("Use delegate_task based dispatch")


class HermesSubAgentClient(BaseLLMClient):
    """通过 Hermes delegate_task 派发标注任务"""

    def __init__(self, name: str, model_override: Optional[str] = None):
        super().__init__(name)
        self.model_override = model_override

    def classify_batch(self, samples, label_system):
        # 由 engine.py 的 _dispatch_batch 方法处理
        raise NotImplementedError("Use engine.py dispatch")


class LLMClientRegistry:
    """LLM 客户端注册表"""

    _clients: Dict[str, BaseLLMClient] = {}

    @classmethod
    def register(cls, client: BaseLLMClient):
        cls._clients[client.name] = client

    @classmethod
    def get(cls, name: str) -> Optional[BaseLLMClient]:
        return cls._clients.get(name)

    @classmethod
    def list_clients(cls) -> List[str]:
        return list(cls._clients.keys())

    @classmethod
    def get_all(cls) -> Dict[str, BaseLLMClient]:
        return dict(cls._clients)

    @classmethod
    def clear(cls):
        cls._clients = {}

    @classmethod
    def register_preset_hermes_models(cls):
        """注册预设的 Hermes 可用模型"""
        # Hermes 通过 config 支持多模型
        # 这里注册元信息，实际调用通过 delegate_task
        models = [
            ("DeepSeek V4 Flash", "deepseek-v4-flash"),
            ("DeepSeek V4", "deepseek-chat"),
            ("Claude Sonnet 4", "anthropic/claude-sonnet-4"),
            ("GPT-4o", "openai/gpt-4o"),
            ("Gemini 2.0 Flash", "google/gemini-2.0-flash-001"),
        ]
        for name, model in models:
            cls._clients[name] = HermesSubAgentClient(name, model)
