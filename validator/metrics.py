"""
Agreement Metrics — 标注一致性评估指标
"""

import math
from collections import defaultdict
from typing import Dict, List


class AgreementMetrics:
    """标注一致性评估指标"""

    @staticmethod
    def cohen_kappa(ann1: Dict[int, str], ann2: Dict[int, str]) -> float:
        """Cohen's Kappa — 两个标注者的一致性"""
        shared = set(ann1.keys()) & set(ann2.keys())
        if not shared:
            return 0.0

        labels = sorted(set(list(ann1.values()) + list(ann2.values())))
        n = len(shared)

        # 混淆矩阵
        confusion = defaultdict(lambda: defaultdict(int))
        for idx in shared:
            confusion[ann1[idx]][ann2[idx]] += 1

        # observed agreement
        po = sum(confusion[l][l] for l in labels) / n

        # expected agreement
        pe = 0
        for l in labels:
            row_sum = sum(confusion[l].values())
            col_sum = sum(confusion[li][l] for li in labels)
            pe += (row_sum * col_sum) / (n * n)

        if pe == 1:
            return 0.0

        kappa = (po - pe) / (1 - pe)
        return round(kappa, 4)

    @staticmethod
    def fleiss_kappa(annotations: Dict[str, Dict[int, str]]) -> float:
        """Fleiss' Kappa — 多个标注者的一致性"""
        llm_names = list(annotations.keys())
        if len(llm_names) < 2:
            return 0.0

        # 收集所有样本
        all_indices = set()
        for ann in annotations.values():
            all_indices.update(ann.keys())

        # 收集所有标签
        all_labels = set()
        for ann in annotations.values():
            all_labels.update(ann.values())
        labels = sorted(all_labels)
        n_llm = len(llm_names)
        n_subjects = len(all_indices)
        n_categories = len(labels)

        # 构建评分矩阵
        # subjects × categories = 每个样本被标为各类别的LLM数
        rating_matrix = []
        for idx in sorted(all_indices):
            row = [0] * n_categories
            for llm in llm_names:
                if idx in annotations[llm]:
                    label = annotations[llm][idx]
                    if label in labels:
                        row[labels.index(label)] += 1
            rating_matrix.append(row)

        if n_subjects == 0:
            return 0.0

        # Pi (每个样本的一致度)
        pis = []
        for row in rating_matrix:
            s = sum(row)
            if s <= 1:
                pis.append(0.0)
                continue
            sum_sq = sum(nij * (nij - 1) for nij in row)
            pi = sum_sq / (s * (s - 1)) if s > 1 else 0
            pis.append(pi)

        p_bar = sum(pis) / n_subjects

        # Pj (每个标签被选中的比例)
        pjs = []
        for j in range(n_categories):
            n_j = sum(row[j] for row in rating_matrix)
            pj = n_j / (n_subjects * n_llm)
            pjs.append(pj)

        p_e_bar = sum(pj * pj for pj in pjs)

        if p_e_bar >= 1:
            return 0.0

        kappa = (p_bar - p_e_bar) / (1 - p_e_bar)
        return round(kappa, 4)

    @staticmethod
    def percentage_agreement(annotations: Dict[str, Dict[int, str]]) -> Dict:
        """总体百分比一致率

        Returns:
            {
                "overall": float,  # 所有LLM全一致的比例
                "majority": float,  # 多数一致的比例
                "per_pair": {llm_a: {llm_b: float}}
            }
        """
        llm_names = list(annotations.keys())
        if len(llm_names) < 2:
            return {"overall": 0, "majority": 0, "per_pair": {}}

        all_indices = set()
        for ann in annotations.values():
            all_indices.update(ann.keys())

        # Per-pair
        per_pair = {}
        for i, la in enumerate(llm_names):
            per_pair[la] = {}
            for j, lb in enumerate(llm_names):
                if i >= j:
                    continue
                shared = set(annotations[la].keys()) & set(annotations[lb].keys())
                agreed = sum(1 for idx in shared if annotations[la][idx] == annotations[lb][idx])
                per_pair[la][lb] = agreed / len(shared) if shared else 0

        # Overall agreement (所有LLM全一致)
        overall_agreed = 0
        for idx in all_indices:
            labels = set()
            for llm in llm_names:
                if idx in annotations[llm]:
                    labels.add(annotations[llm][idx])
            if len(labels) == 1:
                overall_agreed += 1

        # Majority agreement (>=半数一致)
        majority_agreed = 0
        half = len(llm_names) / 2
        for idx in all_indices:
            votes = defaultdict(int)
            for llm in llm_names:
                if idx in annotations[llm]:
                    votes[annotations[llm][idx]] += 1
            if votes and max(votes.values()) >= math.ceil(half):
                majority_agreed += 1

        total = len(all_indices)
        return {
            "overall": overall_agreed / total if total else 0,
            "majority": majority_agreed / total if total else 0,
            "per_pair": per_pair,
        }

    @staticmethod
    def per_label_breakdown(
        annotations: Dict[str, Dict[int, str]], gold_labels: Dict[int, str]
    ) -> Dict:
        """按标签层级 (S1/S2/S3) 计算每类准确率"""
        hierarchy = {"S1": [], "S2": [], "S3": []}

        # 映射标签到层级
        def get_level(label: str) -> str:
            if label.startswith("1."):
                return "S1"
            elif label.startswith("2."):
                return "S2"
            elif label.startswith("3."):
                return "S3"
            return "S1"

        for llm, items in annotations.items():
            level_stats = {level: {"correct": 0, "total": 0} for level in hierarchy}
            for item in items:
                idx = item["idx"]
                if idx not in gold_labels:
                    continue
                level = get_level(gold_labels[idx])
                level_stats[level]["total"] += 1
                if item["label"] == gold_labels[idx]:
                    level_stats[level]["correct"] += 1

            hierarchy[llm + "_by_level"] = {
                level: {
                    "accuracy": stats["correct"] / stats["total"] if stats["total"] else 0,
                    **stats,
                }
                for level, stats in level_stats.items()
            }

        # 标签分布
        label_dist = defaultdict(int)
        for label in gold_labels.values():
            label_dist[label] += 1

        return {
            "label_distribution": dict(label_dist),
            "level_distribution": {
                level: sum(1 for l in gold_labels.values() if get_level(l) == level)
                for level in hierarchy
            },
        }
