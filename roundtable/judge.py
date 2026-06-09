"""外置一致性裁判（蓝图 §4）。

v0 让参与辩论的模型自己说"我们一致了"——模型极易礼貌性附和后宣称收敛，
恰好是要防的东西。判定交给不参与辩论的独立裁判，只做这一件事，
输出结构化裁决。

骨架用确定性启发式裁判（HeuristicJudge 由 Judge 默认实现）：
比对各方 RoundSummary 的立场是否趋同。真实部署应换成独立部署、
低 temperature、结构化输出约束的裁判模型（§10.2），但保持同一接口。
"""

from __future__ import annotations

from typing import Dict, List

from .messages import ConvergenceVerdict, RoundSummary


class Judge:
    """裁判：只读各方 RoundSummary，不发表观点，不参与下一轮。"""

    def __init__(self, agreement_threshold: float = 0.66) -> None:
        # 立场趋同比例 >= 阈值即判收敛。
        self.agreement_threshold = agreement_threshold

    def verdict(self, summaries: List[RoundSummary]) -> ConvergenceVerdict:
        if not summaries:
            return ConvergenceVerdict(converged=False, confidence=0.0)

        # 收敛信号：明确表达"我接受 @X"的让步占比。
        concessions = [s for s in summaries if "我接受" in s.stance or "接受 @" in s.stance]
        concede_ratio = len(concessions) / len(summaries)

        # 立场聚类：归一化后相同立场文本的最大簇占比。
        buckets: Dict[str, int] = {}
        for s in summaries:
            key = _normalize(s.stance)
            buckets[key] = buckets.get(key, 0) + 1
        max_cluster_ratio = max(buckets.values()) / len(summaries)

        signal = max(concede_ratio, max_cluster_ratio)
        converged = signal >= self.agreement_threshold

        agreement_points: List[str] = []
        open_disputes: List[dict] = []
        if converged:
            # 取最大簇的立场作为达成的一致点。
            top_key = max(buckets, key=buckets.get)
            agreement_points = [
                s.stance for s in summaries if _normalize(s.stance) == top_key
            ][:1]
        else:
            open_disputes.append(
                {
                    "topic": "核心立场分歧",
                    "positions": {s.agent_id: s.stance for s in summaries},
                }
            )

        return ConvergenceVerdict(
            converged=converged,
            agreement_points=agreement_points,
            open_disputes=open_disputes,
            confidence=round(signal, 3),
        )


def _normalize(text: str) -> str:
    return "".join(text.lower().split())
