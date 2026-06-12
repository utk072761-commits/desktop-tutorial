"""Decision Board → 分歧矩阵（蓝图 §5）。

不是"共识摘要"：谁来写共识 = 隐形中书令，会污染呈现；强行总结成共识
还会抹平分歧——而分歧恰是裁决最需要看的。这里把裁判裁决 + 各方轮摘要
组装成 DisputeMatrix，呈给用户各方最强版本的对立。
"""

from __future__ import annotations

from typing import Dict, List

from .messages import (
    ConvergenceVerdict,
    Dispute,
    DisputeMatrix,
    RoundSummary,
)


def build_dispute_matrix(
    verdict: ConvergenceVerdict,
    summaries: List[RoundSummary],
    unexplored: List[str] | None = None,
) -> DisputeMatrix:
    """由裁判裁决 + 各方 RoundSummary 组装分歧矩阵（§5）。

    - agreement：取裁判认定的一致点；
    - disputes：每个未决议题列出各方立场 + 最强论据原话；
    - unexplored：尚未触及但相关的点（由调用方/裁判提供，可空）。
    """
    by_id: Dict[str, RoundSummary] = {s.agent_id: s for s in summaries}

    disputes: List[Dispute] = []
    for od in verdict.open_disputes:
        positions: Dict[str, str] = od.get("positions", {})
        strongest: Dict[str, str] = {}
        for aid in positions:
            s = by_id.get(aid)
            if s and s.key_arguments:
                strongest[aid] = s.key_arguments[0]  # 各方最强论据原话
        disputes.append(
            Dispute(
                topic=od.get("topic", "未命名分歧"),
                positions=dict(positions),
                strongest_case=strongest,
            )
        )

    return DisputeMatrix(
        agreement=list(verdict.agreement_points),
        disputes=disputes,
        unexplored=list(unexplored or []),
    )


def render_board(matrix: DisputeMatrix) -> str:
    """把分歧矩阵渲染成三栏文本（§7.2：已一致 / 分歧 / 未探及）。"""
    lines: List[str] = ["=== 决策摘要板 (DisputeMatrix) ==="]

    lines.append("\n【已一致】")
    if matrix.agreement:
        lines.extend(f"  ✓ {a}" for a in matrix.agreement)
    else:
        lines.append("  （无）")

    lines.append("\n【分歧 · 各方最强论点对立】")
    if not matrix.disputes:
        lines.append("  （无）")
    for d in matrix.disputes:
        lines.append(f"  ◆ 议题: {d.topic}")
        for aid, stance in d.positions.items():
            case = d.strongest_case.get(aid, "—")
            lines.append(f"      [{aid}] 立场: {stance}")
            lines.append(f"             最强论据: {case}")

    lines.append("\n【未探及】")
    if matrix.unexplored:
        lines.extend(f"  · {u}" for u in matrix.unexplored)
    else:
        lines.append("  （无）")

    return "\n".join(lines)
