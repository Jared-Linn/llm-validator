"""
Consensus Engine — 多LLM标注共识判定
思路B核心：多个LLM独立标注 → 投票/加权 → 最终标签
"""

import json
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple


class ConsensusEngine:
    """多LLM共识引擎"""

    def __init__(self, min_agreement: int = 2, method: str = "majority"):
        """
        Args:
            min_agreement: 达成共识所需的最少LLM同意数
            method: 'majority' | 'weighted' | 'confidence'
        """
        self.min_agreement = min_agreement
        self.method = method
        self.llm_weights: Dict[str, float] = {}  # LLM名称 → 权重

    def set_llm_weights(self, weights: Dict[str, float]):
        """为每个LLM设置权重（基于历史表现）"""
        self.llm_weights = weights

    def compute_consensus(
        self, annotations: Dict[str, List[Dict]]
    ) -> Dict:
        """对一批样本计算多LLM共识

        Args:
            annotations: {llm_name: [{"idx": int, "label": str, "confidence": float}, ...]}

        Returns:
            {
                "sample_idx": {
                    "majority_label": str,
                    "agreement_count": int,
                    "total_voters": int,
                    "agreement_ratio": float,
                    "all_labels": {label: count|weighted_sum},
                    "llm_votes": {llm_name: label},
                    "is_consensus": bool,
                    "max_label": str
                }
            }
        """
        if not annotations:
            return {}

        # 收集所有 LLM 名称
        llm_names = list(annotations.keys())
        n_llms = len(llm_names)

        # 按 idx 索引各 LLM 的标注
        idx_to_votes: Dict[int, Dict] = defaultdict(dict)
        idx_to_conf: Dict[int, Dict] = defaultdict(dict)

        for llm in llm_names:
            for item in annotations[llm]:
                idx = item["idx"]
                label = item["label"]
                idx_to_votes[idx][llm] = label
                idx_to_conf[idx][llm] = item.get("confidence", 1.0)

        results = {}
        for idx, votes in idx_to_votes.items():
            labels = list(votes.values())
            counter = Counter(labels)

            if self.method == "majority":
                # 简单多数投票
                max_count = counter.most_common(1)[0][1]
                max_label = counter.most_common(1)[0][0]

                # 平局处理：取置信度加权和最高的标签
                if len([c for c in counter.values() if c == max_count]) > 1:
                    # 有平局 → 加权投票
                    label_scores = defaultdict(float)
                    for llm in llm_names:
                        if llm in votes:
                            label_scores[votes[llm]] += idx_to_conf[idx].get(llm, 1.0)
                    max_label = max(label_scores, key=label_scores.get)
                    agreement_count = sum(
                        1 for l in labels if l == max_label
                    )
                else:
                    agreement_count = max_count

            elif self.method == "weighted":
                # 加权投票 (基于LLM历史权重)
                label_scores = defaultdict(float)
                for llm in llm_names:
                    if llm in votes:
                        weight = self.llm_weights.get(llm, 1.0)
                        label_scores[votes[llm]] += weight * idx_to_conf[idx].get(llm, 1.0)
                max_label = max(label_scores, key=label_scores.get)
                agreement_count = sum(
                    1 for l in labels if l == max_label
                )

            else:
                # confidence: 直接取置信度最高的LLM的标签
                best_llm = max(
                    [llm for llm in llm_names if llm in votes],
                    key=lambda llm: idx_to_conf[idx].get(llm, 0),
                )
                max_label = votes[best_llm]
                agreement_count = sum(1 for l in labels if l == max_label)

            results[str(idx)] = {
                "majority_label": max_label,
                "agreement_count": agreement_count,
                "total_voters": len(votes),
                "agreement_ratio": agreement_count / len(votes) if votes else 0,
                "all_labels": dict(counter),
                "llm_votes": dict(votes),
                "is_consensus": agreement_count >= min(self.min_agreement, len(votes)),
                "max_label": max_label,
            }

        return results

    def compute_llm_pairwise_agreement(
        self, annotations: Dict[str, List[Dict]]
    ) -> Dict[str, Dict]:
        """计算每对LLM之间的一致率

        Returns:
            {llm_a: {llm_b: {"agreed": N, "total": N, "ratio": float}}}
        """
        llm_names = list(annotations.keys())
        pairwise = {}

        # 索引
        idx_by_llm = {}
        for llm in llm_names:
            idx_by_llm[llm] = {item["idx"]: item["label"] for item in annotations[llm]}

        for i, llm_a in enumerate(llm_names):
            pairwise[llm_a] = {}
            for j, llm_b in enumerate(llm_names):
                if i == j:
                    pairwise[llm_a][llm_b] = {"agreed": 0, "total": 0, "ratio": 1.0}
                    continue

                shared = set(idx_by_llm[llm_a].keys()) & set(idx_by_llm[llm_b].keys())
                agreed = sum(
                    1 for idx in shared if idx_by_llm[llm_a][idx] == idx_by_llm[llm_b][idx]
                )
                total = len(shared)
                pairwise[llm_a][llm_b] = {
                    "agreed": agreed,
                    "total": total,
                    "ratio": agreed / total if total > 0 else 0,
                }

        return pairwise

    def compute_llm_vs_gold(
        self, annotations: Dict[str, List[Dict]], gold_labels: Dict[int, str]
    ) -> Dict[str, Dict]:
        """每个LLM vs Gold Standard 的准确率

        Returns:
            {llm_name: {"correct": N, "total": N, "accuracy": float}}
        """
        results = {}
        for llm, items in annotations.items():
            correct = 0
            total = 0
            for item in items:
                idx = item["idx"]
                if idx in gold_labels:
                    total += 1
                    if item["label"] == gold_labels[idx]:
                        correct += 1
            results[llm] = {
                "correct": correct,
                "total": total,
                "accuracy": correct / total if total > 0 else 0,
            }
        return results
