"""
Annotation Validator — 多LLM验证引擎
思路B核心实现：加载数据 → 采样 → 多LLM验证 → 共识判定 → 评估
"""

import json
import math
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

from .consensus import ConsensusEngine
from .llm_clients import LLMClientRegistry
from .metrics import AgreementMetrics


SAMPLE_JSON_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "sample_data.json"
)


class AnnotationValidator:
    """多LLM标注验证器"""

    def __init__(self):
        self.consensus = ConsensusEngine()
        self.metrics = AgreementMetrics()
        self.results_cache = {}

    def load_data(self, json_path: str) -> List[Dict]:
        """加载标注数据

        Args:
            json_path: 标注JSON文件路径 (格式如 student-XX_labeled_refined.json)

        Returns:
            [{title, content, labels, ...}]
        """
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    def sample_uncertain(
        self, data: List[Dict], n: int = 200, method: str = "random"
    ) -> List[Dict]:
        """采样不确定/代表性样本

        Args:
            data: 全量标注数据
            n: 采样数量
            method: 'random' | 'stratified' (按S1/S2/S3分层)
        """
        if method == "stratified":
            # 分层抽样
            buckets = {"S1": [], "S2": [], "S3": []}
            for item in data:
                label = item.get("labels", {}).get("label", "1.0")
                if label.startswith("3."):
                    buckets["S3"].append(item)
                elif label.startswith("2."):
                    buckets["S2"].append(item)
                else:
                    buckets["S1"].append(item)

            sampled = []
            per_level = max(1, n // 3)
            for level in ["S1", "S2", "S3"]:
                pool = buckets[level]
                size = min(per_level, len(pool))
                sampled.extend(random.sample(pool, size))

            # 补足不足的
            remaining = n - len(sampled)
            if remaining > 0:
                all_indices = set(id(item) for item in sampled)
                rest = [item for item in data if id(item) not in all_indices]
                sampled.extend(random.sample(rest, min(remaining, len(rest))))

            random.shuffle(sampled)
            return sampled[:n]
        else:
            return random.sample(data, min(n, len(data)))

    def prepare_validation_set(
        self, data: List[Dict], n: int = 200, method: str = "stratified"
    ) -> Tuple[List[Dict], Dict[int, str]]:
        """准备验证集

        Returns:
            (samples, gold_labels)
            - samples: 待验证的样本列表
            - gold_labels: {idx: label} — 用原标注作为基准
        """
        samples = self.sample_uncertain(data, n, method)
        gold_labels = {}
        for i, item in enumerate(samples):
            idx = i  # 使用样本在列表中的索引
            gold_labels[idx] = item.get("labels", {}).get("label", "1.0")

        # 给样本添加 idx 字段
        for i, item in enumerate(samples):
            item["_idx"] = i

        return samples, gold_labels

    def _build_label_prompt(self, label_system: Optional[str] = None) -> str:
        """构建标签体系提示词"""
        if label_system:
            return label_system

        return """
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
请只输出格式: idx → 标签
例如: 0 → 1.7
"""

    def _build_sample_text(self, sample: Dict) -> str:
        """构建单个样本的展示文本"""
        title = sample.get("title", "")
        content_raw = sample.get("content", "")
        # 截取前500个字符作为内容摘要
        content = content_raw[:500] if isinstance(content_raw, str) else str(content_raw)
        dialog = sample.get("dialog", [])
        dialog_text = ""
        if dialog:
            turns = dialog[:6]  # 最多取6轮对话
            for turn in turns:
                speaker = turn.get("speaker", turn.get("role", "用户"))
                text = turn.get("text", turn.get("content", ""))[:200]
                dialog_text += f"[{speaker}]: {text}\n"

        return f"标题: {title}\n内容: {content}\n对话:\n{dialog_text}"

    def _dispatch_to_llm(
        self, llm_name: str, samples: List[Dict], label_prompt: str
    ) -> List[Dict]:
        """派发标注任务到单一LLM（通过delegate_task子代理）"""
        # 构建任务数据
        task_items = []
        for s in samples:
            task_items.append({
                "idx": s["_idx"],
                "title": s.get("title", ""),
                "content": s.get("content", "")[:500],
                "dialog": s.get("dialog", [])[:6],
            })

        # 构建提交给子代理的任务JSON文件
        import tempfile
        task_path = os.path.join(tempfile.gettempdir(), f"llmv_task_{llm_name.replace(' ','_')}.json")
        with open(task_path, "w", encoding="utf-8") as f:
            json.dump(task_items, f, ensure_ascii=False, indent=2)

        return task_items  # 返回任务描述（实际标注通过子代理在外部完成）

    def run_validation(
        self,
        data_path: str,
        n_samples: int = 200,
        sampling_method: str = "stratified",
        label_system: Optional[str] = None,
        llm_list: Optional[List[str]] = None,
    ) -> Dict:
        """运行多LLM验证

        Args:
            data_path: 标注数据JSON路径
            n_samples: 采样数量
            sampling_method: 采样方法
            label_system: 自定义标签体系
            llm_list: 参与验证的LLM列表

        Returns:
            验证结果字典（包含所有统计数据）
        """
        # 1. 加载数据
        data = self.load_data(data_path)

        # 2. 准备验证集
        samples, gold_labels = self.prepare_validation_set(data, n_samples, sampling_method)

        # 3. 获取LLM列表
        if not llm_list:
            LLMClientRegistry.register_preset_hermes_models()
            llm_list = LLMClientRegistry.list_clients()

        # 4. 构建标签提示
        prompt = self._build_label_prompt(label_system)

        # 5. 为每个LLM准备任务（生成任务文件）
        task_results = {}
        for llm in llm_list:
            task_results[llm] = self._dispatch_to_llm(llm, samples, prompt)

        # 6. 返回验证结构（含待标注的任务）
        result = {
            "status": "ready",
            "samples_count": len(samples),
            "llm_count": len(llm_list),
            "llm_list": llm_list,
            "sampling_method": sampling_method,
            "gold_labels": {str(k): v for k, v in gold_labels.items()},
            "samples": [
                {
                    "idx": s["_idx"],
                    "title": s.get("title", "")[:80],
                    "content": self._build_sample_text(s),
                    "gold_label": gold_labels.get(s["_idx"], "?"),
                }
                for s in samples
            ],
            "label_system": prompt,
            "llm_tasks": task_results,
        }

        self.results_cache = result
        return result

    def ingest_llm_results(
        self,
        llm_name: str,
        results: List[Dict],
    ) -> Dict:
        """导入某个LLM的标注结果

        Args:
            llm_name: LLM名称
            results: [{"idx": int, "label": str, "confidence": float, "reasoning": str}]

        Returns:
            更新后的验证摘要
        """
        if "llm_annotations" not in self.results_cache:
            self.results_cache["llm_annotations"] = {}

        self.results_cache["llm_annotations"][llm_name] = results
        return self.compute_summary()

    def compute_summary(self) -> Dict:
        """计算当前所有结果的摘要统计"""
        cache = self.results_cache
        if "llm_annotations" not in cache or not cache["llm_annotations"]:
            return {"status": "incomplete", "message": "尚无LLM标注结果"}

        annotations = cache["llm_annotations"]
        gold_labels_raw = cache.get("gold_labels", {})

        # 转换为 metrics 需要的格式
        ann_dict: Dict[str, Dict[int, str]] = {}
        ann_items: Dict[str, List[Dict]] = {}
        for llm, items in annotations.items():
            ann_dict[llm] = {item["idx"]: item["label"] for item in items}
            ann_items[llm] = items

        gold_int = {int(k): v for k, v in gold_labels_raw.items()}

        # 共识计算
        consensus_results = self.consensus.compute_consensus(ann_items)

        # 生成 consensus 标签作为伪金标准
        consensus_labels = {}
        for idx_str, cr in consensus_results.items():
            consensus_labels[int(idx_str)] = cr["majority_label"]

        # 指标计算
        pairwise = self.consensus.compute_llm_pairwise_agreement(ann_items)
        vs_gold = self.consensus.compute_llm_vs_gold(ann_items, gold_int)
        vs_consensus = self.consensus.compute_llm_vs_gold(ann_items, consensus_labels)

        # Fleiss Kappa
        fleiss = self.metrics.fleiss_kappa(ann_dict)

        # Percentage agreement
        pct_agree = self.metrics.percentage_agreement(ann_dict)

        # Per-label breakdown
        label_breakdown = self.metrics.per_label_breakdown(ann_items, gold_int)

        # 构建摘要
        summary = {
            "status": "completed",
            "total_samples": cache.get("samples_count", 0),
            "llm_count": len(annotations),
            "llm_list": list(annotations.keys()),
            "fleiss_kappa": fleiss,
            "percentage_agreement": pct_agree,
            "pairwise_agreement": pairwise,
            "llm_vs_gold": vs_gold,
            "llm_vs_consensus": vs_consensus,
            "consensus_results": consensus_results,
            "consensus_labels": {str(k): v for k, v in consensus_labels.items()},
            "gold_labels": gold_labels_raw,
            "label_breakdown": label_breakdown,
            "sampling_method": cache.get("sampling_method", "unknown"),
        }

        # 汇总准确率排名
        accuracies = []
        for llm, stats in vs_gold.items():
            accuracies.append({"llm": llm, "accuracy": stats["accuracy"]})
        accuracies.sort(key=lambda x: x["accuracy"], reverse=True)
        summary["llm_ranking"] = accuracies

        # 共识 vs 原始标签 一致率
        consensus_vs_gold_agree = sum(
            1 for k, v in consensus_labels.items()
            if k in gold_int and v == gold_int[k]
        )
        consensus_vs_gold_total = sum(
            1 for k in consensus_labels if k in gold_int
        )
        summary["consensus_vs_gold"] = {
            "agreed": consensus_vs_gold_agree,
            "total": consensus_vs_gold_total,
            "accuracy": consensus_vs_gold_agree / consensus_vs_gold_total
            if consensus_vs_gold_total else 0,
        }

        self.results_cache["summary"] = summary
        return summary

    def export_json(self, output_path: str):
        """导出完整验证结果到JSON"""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.results_cache, f, ensure_ascii=False, indent=2)

    @staticmethod
    def generate_demo_data(
        data_path: str, output_path: Optional[str] = None
    ):
        """从标注数据生成演示用的验证集样本（供看板展示）"""
        import random

        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 模拟3个LLM的标注结果
        random.seed(42)
        # 采样200条
        samples = random.sample(data, min(200, len(data)))
        sample_info = []
        for i, s in enumerate(samples):
            label = s.get("labels", {}).get("label", "1.0")
            sample_info.append({
                "idx": i,
                "title": s.get("title", "")[:60],
                "content": str(s.get("content", ""))[:300],
                "gold_label": label,
            })

        # 模拟LLM标注 (基于原始标签加随机扰动)
        llm_names = ["DeepSeek V4 Flash", "Claude Sonnet 4", "Gemini 2.0 Flash"]
        llm_results = {name: [] for name in llm_names}

        for i, s in enumerate(sample_info):
            gold = s["gold_label"]
            for llm in llm_names:
                # 模拟准确率：DeepSeek ~85%, Claude ~82%, Gemini ~78%
                if llm == "DeepSeek V4 Flash":
                    accuracy = 0.85
                elif llm == "Claude Sonnet 4":
                    accuracy = 0.82
                else:
                    accuracy = 0.78

                if random.random() < accuracy:
                    label = gold
                    confidence = round(0.75 + random.random() * 0.2, 2)
                else:
                    # 随机选一个其他标签
                    all_labels = ["1.1", "1.2", "1.3", "1.4", "1.5", "1.6", "1.7",
                                  "1.8", "1.9", "1.10", "2.1", "2.2", "2.3", "2.4",
                                  "2.5", "3.1", "3.2", "3.3"]
                    others = [l for l in all_labels if l != gold]
                    label = random.choice(others)
                    confidence = round(0.3 + random.random() * 0.3, 2)

                llm_results[llm].append({
                    "idx": i,
                    "label": label,
                    "confidence": confidence,
                })

        demo_data = {
            "samples": sample_info,
            "llm_names": llm_names,
            "llm_annotations": llm_results,
            "gold_labels": {str(i): s["gold_label"] for i, s in enumerate(sample_info)},
            "total_samples": len(sample_info),
            "generated_at": "demo",
        }

        out = output_path or SAMPLE_JSON_PATH
        with open(out, "w", encoding="utf-8") as f:
            json.dump(demo_data, f, ensure_ascii=False, indent=2)

        return out
