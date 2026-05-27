"""
LLM Providers — 多LLM真实API接入
支持：OpenAI / Anthropic / DeepSeek / Gemini / 自定义
"""

import asyncio
import json
import os
import random
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import httpx

# ── 标签体系提示词（供所有 LLM 使用） ──

LABEL_SYSTEM = """
你是一名专业的心理咨询对话分类专家。请对以下对话内容进行分类。

分类体系（三级标签）：

【S1 — 日常困扰（轻度心理不适）】
1.1 学业问题 | 1.2 职场工作 | 1.3 家庭矛盾 | 1.4 婚恋情感
1.5 人际关系 | 1.6 失眠睡眠 | 1.7 压力管理 | 1.8 情绪调节
1.9 自我成长 | 1.10 性格困扰 | 1.11 社交恐惧 | 1.12 考试焦虑
1.13 厌学情绪 | 1.14 亲子教育 | 1.15 经济压力 | 1.16 人生迷茫
1.17 其他日常困扰

【S2 — 中度心理障碍】
2.1 抑郁症 | 2.2 焦虑症 | 2.3 双相情感障碍 | 2.4 创伤后应激(PTSD)
2.5 饮食障碍 | 2.6 强迫症(OCD) | 2.7 恐惧症 | 2.8 躯体化障碍
2.9 其他心理障碍

【S3 — 紧急危机】
3.1 正在自杀 | 3.2 自杀计划/意图 | 3.3 自残行为 | 3.4 伤害他人
3.5 报复社会/他人

优先级：S3 > S2 > S1（如果对话涉及多个层级，取最高级）
请只输出 JSON 数组格式，每条含 idx 和 label 字段。
示例：[{"idx": 0, "label": "1.7", "confidence": 0.92, "reasoning": "..."}]
"""


# ── 基类 ──

class BaseLLMProvider(ABC):
    """LLM 提供商基类"""

    def __init__(self, name: str, model: str, api_key: str, base_url: str):
        self.name = name
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    @abstractmethod
    async def classify_batch(self, samples: List[Dict]) -> List[Dict]:
        """对一批样本进行分类

        Args:
            samples: [{"idx": int, "title": str, "content": str}, ...]

        Returns:
            [{"idx": int, "label": str, "confidence": float, "reasoning": str}]
        """
        ...

    def _build_prompt(self, samples: List[Dict]) -> str:
        """构建分类提示词"""
        items = []
        for s in samples:
            title = (s.get("title") or "")[:100]
            content = (s.get("content") or "")[:300]
            items.append(f"[条目 {s['idx']}]\n标题: {title}\n内容: {content}")

        return f"{LABEL_SYSTEM}\n\n请对以下 {len(samples)} 条对话进行分类:\n\n" + "\n\n".join(items)


# ── OpenAI 兼容 (OpenAI / DeepSeek / 自定义) ──

class OpenAICompatibleProvider(BaseLLMProvider):
    """OpenAI 兼容 API (支持 OpenAI, DeepSeek, 自定义端点)"""

    async def classify_batch(self, samples: List[Dict]) -> List[Dict]:
        prompt = self._build_prompt(samples)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "你是一名专业的心理咨询对话分类专家。"},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 4096,
        }

        url = f"{self.base_url}/chat/completions"
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

        content = data["choices"][0]["message"]["content"]
        return self._parse_response(content, samples)

    def _parse_response(self, content: str, samples: List[Dict]) -> List[Dict]:
        """解析 LLM 返回的 JSON"""
        # 尝试解析 JSON
        content = content.strip()
        # 去掉可能的 markdown 代码块标记
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1])

        try:
            results = json.loads(content)
            if isinstance(results, list):
                return results
        except json.JSONDecodeError:
            pass

        # 如果解析失败，从文本中提取 idx → label
        results = []
        import re
        pattern = re.compile(r'(?:idx|条目)\s*[:：]?\s*(\d+).*?label\s*[:：]?\s*["\']?(\d+\.\d+)', re.DOTALL)
        for s in samples:
            idx_str = str(s["idx"])
            # 尝试在 content 中找到对应条目
            match = re.search(rf'(?:idx|条目)\s*[:：]?\s*{idx_str}\s*.*?label\s*[:：]?\s*["\']?(\d+\.\d+)', content)
            if match:
                label = match.group(1)
                results.append({"idx": s["idx"], "label": label, "confidence": 0.8, "reasoning": ""})
            else:
                results.append({"idx": s["idx"], "label": "1.17", "confidence": 0.5, "reasoning": "parse_fallback"})

        return results if results else [
            {"idx": s["idx"], "label": "1.17", "confidence": 0.5, "reasoning": "parse_failed"}
            for s in samples
        ]


# ── Anthropic ──

class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude API"""

    async def classify_batch(self, samples: List[Dict]) -> List[Dict]:
        prompt = self._build_prompt(samples)
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}],
        }

        url = f"{self.base_url}/messages"
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

        content = data["content"][0]["text"]
        return OpenAICompatibleProvider._parse_response_static(content, samples)


# ── Google Gemini ──

class GeminiProvider(BaseLLMProvider):
    """Google Gemini API"""

    async def classify_batch(self, samples: List[Dict]) -> List[Dict]:
        prompt = self._build_prompt(samples)
        url = f"{self.base_url}/models/{self.model}:generateContent?key={self.api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.1, "maxOutputTokens": 4096},
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

        content = data["candidates"][0]["content"]["parts"][0]["text"]
        return OpenAICompatibleProvider._parse_response_static(content, samples)


# ── 免费模拟 (兜底) ──

class FreeSimulatedProvider(BaseLLMProvider):
    """免费内置 — 基于已有标注的模拟验证"""

    def __init__(self):
        super().__init__("Free (Simulated)", "free-simulated", "", "")
        self.seed = 42

    async def classify_batch(self, samples: List[Dict]) -> List[Dict]:
        """基于输入数据中的 gold_label 做轻微扰动模拟"""
        random.seed(self.seed + hash(str(samples)) % 10000)
        results = []
        for s in samples:
            gold = s.get("gold_label", "1.17")
            if random.random() < 0.82:
                label = gold
                conf = round(0.75 + random.random() * 0.2, 2)
            else:
                candidates = [l for l in
                    ["1.1","1.2","1.3","1.4","1.5","1.6","1.7","1.8","1.9",
                     "1.10","1.11","1.12","1.13","1.14","1.15","1.16","1.17",
                     "2.1","2.2","2.3","2.4","2.5","2.6","2.7","2.8","2.9",
                     "3.1","3.2","3.3","3.4","3.5"] if l != gold]
                label = random.choice(candidates)
                conf = round(0.3 + random.random() * 0.3, 2)
            results.append({
                "idx": s["idx"],
                "label": label,
                "confidence": conf,
                "reasoning": "simulated",
            })
        return results


# ── 编排器 ──

PROVIDER_CLASSES = {
    "openai": OpenAICompatibleProvider,
    "deepseek": OpenAICompatibleProvider,
    "custom": OpenAICompatibleProvider,
    "anthropic": AnthropicProvider,
    "gemini": GeminiProvider,
    "free": FreeSimulatedProvider,
}


class LLMOrchestrator:
    """LLM 编排器 — 根据用户配置选择可用的 LLM 并行验证"""

    def __init__(self, user_llm_configs: List[Dict] = None):
        self.user_configs = user_llm_configs or []

    def get_providers(self, samples: List[Dict]) -> List[BaseLLMProvider]:
        """根据用户配置构建可用的 LLM 提供商列表"""
        providers = []

        for cfg in self.user_configs:
            provider_id = cfg["provider"]
            cls = PROVIDER_CLASSES.get(provider_id)
            if not cls:
                continue
            if not cfg.get("api_key") and provider_id != "free":
                continue

            if provider_id == "free":
                p = FreeSimulatedProvider()
            elif provider_id in ("openai", "deepseek", "custom"):
                p = OpenAICompatibleProvider(
                    name=cfg.get("label", provider_id),
                    model=cfg.get("model", ""),
                    api_key=cfg.get("api_key", ""),
                    base_url=cfg.get("base_url", ""),
                )
            elif provider_id == "anthropic":
                p = AnthropicProvider(
                    name="Claude",
                    model=cfg.get("model", "claude-haiku-3-5"),
                    api_key=cfg.get("api_key", ""),
                    base_url=cfg.get("base_url", "https://api.anthropic.com/v1"),
                )
            elif provider_id == "gemini":
                p = GeminiProvider(
                    name="Gemini",
                    model=cfg.get("model", "gemini-2.0-flash"),
                    api_key=cfg.get("api_key", ""),
                    base_url=cfg.get("base_url", "https://generativelanguage.googleapis.com/v1beta"),
                )
            else:
                continue

            # 注入 gold_label 给 free provider 模拟用
            if isinstance(p, FreeSimulatedProvider):
                golds = {s["idx"]: s.get("gold_label", "1.17") for s in samples}
                # 将 golds 作为样本属性
                augmented = []
                for s in samples:
                    s_copy = dict(s)
                    s_copy["gold_label"] = golds.get(s["idx"], "1.17")
                    augmented.append(s_copy)
                p.recent_samples = augmented

            providers.append(p)

        # 如果没有配置任何 LLM，使用免费模拟
        if not providers:
            providers.append(FreeSimulatedProvider())

        return providers

    async def run_parallel_validation(
        self, samples: List[Dict]
    ) -> Dict[str, List[Dict]]:
        """并行调用所有可用的 LLM 进行验证

        Returns:
            {provider_name: [{"idx": int, "label": str, "confidence": float}, ...]}
        """
        providers = self.get_providers(samples)
        results = {}

        # 分批：每批 20 条
        batch_size = 20
        for provider in providers:
            provider_results = []
            for i in range(0, len(samples), batch_size):
                batch = samples[i:i + batch_size]
                try:
                    batch_results = await provider.classify_batch(batch)
                    provider_results.extend(batch_results)
                except Exception as e:
                    # 失败时 fallback
                    provider_results.extend([
                        {"idx": s["idx"], "label": "1.17", "confidence": 0.5,
                         "reasoning": f"error: {str(e)[:50]}"}
                        for s in batch
                    ])
            results[provider.name] = provider_results

        return results


# Parse response 静态方法
def _parse_response_static(content: str, samples: List[Dict]) -> List[Dict]:
    return OpenAICompatibleProvider._parse_response_static(content, samples)


OpenAICompatibleProvider._parse_response_static = staticmethod(_parse_response_static)
