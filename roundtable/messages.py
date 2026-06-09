"""内部统一数据结构（蓝图 §2.1 / §3.2 / §4 / §5）。

所有模型对外只见内部格式，provider 差异封死在 adapter 内（§2.1）。
另含辩论流转过程中的几个结构化载体：轮摘要、裁判裁决、分歧矩阵。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List

# 合法的发言角色与系统保留 agent_id。
ROLES = ("user", "agent", "moderator", "system")
RESERVED_IDS = ("user", "judge", "moderator", "system")


@dataclass
class InternalMessage:
    """圆桌内部统一消息格式（§2.1）。

    role:     "user" | "agent" | "moderator" | "system"
    agent_id: "A1" | "A2" | "A3" | "user" | "judge" ...
    content:  发言正文（含末尾结构化字段，见 §6）
    refs:     本条发言 @ 引用/反驳的 agent_id，渲染连线用
    round:    所属辩论轮次，0 = 用户原始指令
    """

    role: str
    agent_id: str
    content: str
    refs: List[str] = field(default_factory=list)
    round: int = 0
    # tool_use 轨迹（本条发言前模型调用了哪些工具，供审计/UI 渲染）；
    # 每项 {name, arguments, result, is_error}。普通发言为空。
    tool_trace: List[Dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"非法 role: {self.role!r}，应为 {ROLES}")
        if self.round < 0:
            raise ValueError("round 不可为负")

    # ── 结构化字段解析（§6：发言末尾自带 STANCE/ARGS/REBUTS）──────────
    def parse_structured(self) -> "RoundSummary":
        """从发言正文末尾抽取 STANCE/ARGS/REBUTS，压成 RoundSummary。

        协作协议要求模型在输出末尾自带结构化字段，这样压缩 context
        时可省去一次额外的摘要调用（§3.2 末段）。字段缺失时退化为
        取正文首句作为立场。
        """
        stance = _extract_field(self.content, "STANCE")
        args_raw = _extract_field(self.content, "ARGS")
        rebuts_raw = _extract_field(self.content, "REBUTS")

        if not stance:
            # 退化：取正文第一句作为立场，避免空摘要。
            first = self.content.strip().split("\n", 1)[0].strip()
            stance = first[:200] if first else "(未表态)"

        key_arguments = _split_items(args_raw) if args_raw else []
        # REBUTS 优先用结构化字段，否则回落到已解析的 refs。
        rebuts = _split_items(rebuts_raw) if rebuts_raw else list(self.refs)

        return RoundSummary(
            agent_id=self.agent_id,
            stance=stance,
            key_arguments=key_arguments,
            rebuts=rebuts,
            round=self.round,
        )


@dataclass
class RoundSummary:
    """每轮结束后的结构化摘要（§3.2）。

    下一轮只传 [用户原始问题] + [上一轮各方 RoundSummary]，
    而非滚雪球的原始全文，从而避免 context 线性膨胀。
    """

    agent_id: str
    stance: str
    key_arguments: List[str] = field(default_factory=list)
    rebuts: List[str] = field(default_factory=list)
    round: int = 0

    def render(self) -> str:
        """压缩为喂给下一轮模型的紧凑文本。"""
        args = "; ".join(self.key_arguments) if self.key_arguments else "—"
        rebuts = ", ".join(self.rebuts) if self.rebuts else "—"
        return (
            f"[{self.agent_id}] 立场: {self.stance} | "
            f"论据: {args} | 反驳: {rebuts}"
        )


@dataclass
class ConvergenceVerdict:
    """外置裁判的结构化裁决（§4）。

    裁判只读各方 RoundSummary，不发表观点，不参与下一轮。
    converged=True 或 n>=n_max -> PROPOSAL。
    """

    converged: bool
    agreement_points: List[str] = field(default_factory=list)
    open_disputes: List[Dict] = field(default_factory=list)  # [{topic, positions}]
    confidence: float = 0.0


@dataclass
class Dispute:
    """单个对立点（§5）。"""

    topic: str
    positions: Dict[str, str] = field(default_factory=dict)        # {agent_id: 立场}
    strongest_case: Dict[str, str] = field(default_factory=dict)   # {agent_id: 最强论据原话}


@dataclass
class DisputeMatrix:
    """决策板 = 分歧矩阵，而非共识摘要（§5）。

    呈给用户的是各方最强版本的对立，不是妥协后的平庸中值。
    """

    agreement: List[str] = field(default_factory=list)       # 真正一致的点
    disputes: List[Dispute] = field(default_factory=list)    # 对立点
    unexplored: List[str] = field(default_factory=list)      # 尚未触及但相关的点


# ── 内部辅助 ────────────────────────────────────────────────────────────

_FIELD_RE_TMPL = r"{name}\s*[:：]\s*(?P<val>.+?)(?=\n[A-Z]+\s*[:：]|\Z)"


def _extract_field(text: str, name: str) -> str:
    """抽取形如 `NAME: ...` 的结构化字段（支持中英文冒号、跨行至下个字段）。"""
    m = re.search(_FIELD_RE_TMPL.format(name=name), text, re.DOTALL)
    return m.group("val").strip() if m else ""


def _split_items(raw: str) -> List[str]:
    """把一段论据/反驳文本拆成条目列表。

    支持分隔符：换行、分号、中文顿号、以及 `1.`/`-`/`•` 等条目前缀。
    """
    parts = re.split(r"[\n;；、]|(?:^|\s)[-•]\s|(?:^|\s)\d+[.)]\s", raw)
    return [p.strip(" -•\t") for p in parts if p and p.strip(" -•\t")]
