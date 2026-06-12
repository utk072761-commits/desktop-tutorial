"""辩论仲裁器 The Moderator（蓝图 §3）。

修正 v0 三处：
  - §3.1 并发取代 mutex（互斥串行让延迟累加，不可用）；
  - §3.2 context 压缩（每轮压成 RoundSummary，下一轮只喂摘要 + 原始问题）；
  - §3.3 turn-counter 与 token-budget 职责分离（防死循环 vs 防成本失控）。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

from .adapters import AgentAdapter
from .messages import InternalMessage, RoundSummary
from .tools import Tool, run_tool_loop


class BudgetExceeded(Exception):
    """全局 token 熔断（§3.3 / §10.4）。"""


@dataclass
class TokenLedger:
    """成本账本，熔断依据（§8）。"""

    per_round: List[int] = field(default_factory=list)
    total: int = 0

    def charge(self, tokens: int) -> None:
        self.per_round.append(tokens)
        self.total += tokens


def _estimate_tokens(msg: InternalMessage) -> int:
    """极简 token 估算（骨架用空白切词近似；真实接 provider usage）。"""
    return max(1, len(msg.content.split()))


def order_by_seat(seats: List[str], results: Dict[str, object]) -> List[Tuple[str, object]]:
    """按固定座次排序渲染（§3.1：后端并发，前端按固定座次）。"""
    return [(aid, results[aid]) for aid in seats if aid in results]


class Moderator:
    """主持人：并发跑一轮、压缩 context、看管双闸门。"""

    def __init__(
        self,
        registry: Dict[str, AgentAdapter],
        n_max: int = 5,
        per_round_token_budget: int = 4096,
        global_token_budget: int = 200_000,
    ) -> None:
        self.registry = registry
        self.n_max = n_max
        self.per_round_token_budget = per_round_token_budget
        self.global_token_budget = global_token_budget
        self.ledger = TokenLedger()

    # ── §3.1 并发调用（取代 Mutex）─────────────────────────────────────
    async def run_round(
        self,
        active: Set[str],
        ctx: List[InternalMessage],
        tools: Optional[List[Tool]] = None,
        on_token=None,
    ) -> List[InternalMessage]:
        """并发生成一轮发言；单模型失败隔离为空缺，不拖垮整轮。

        返回值按固定座次排序，渲染层据此稳定布局（不随到达顺序跳动）。
        若提供 tools 且某 adapter 支持 tool_use，则该位走 agentic 工具循环
        （模型→工具调用→结果→终稿），否则走普通流式发言。
        on_token(agent_id, token) 逐 token 回调（§3.1：后端并发、前端渲染分层——
        UI 据此实时呈现，但最终落座仍按固定座次）。tool loop 为非流式，不回调。
        """
        seats = [aid for aid in sorted(self.registry) if aid in active]
        tasks = {aid: self._respond(self.registry[aid], ctx, tools, on_token) for aid in seats}

        gathered = await asyncio.gather(*tasks.values(), return_exceptions=True)
        results: Dict[str, InternalMessage] = {}
        for aid, res in zip(tasks.keys(), gathered):
            if isinstance(res, Exception):
                # 失败隔离：降级为该位空缺（§3.1）。
                continue
            results[aid] = res

        ordered = order_by_seat(seats, results)
        msgs = [m for _, m in ordered]
        self._charge_round(msgs)
        return msgs

    @staticmethod
    def _respond(adapter, ctx, tools, on_token=None):
        """单个 adapter 的本轮发言:有工具且支持则走工具循环,否则普通发言。"""
        if tools and getattr(adapter, "supports_tools", False):
            return run_tool_loop(adapter, ctx, tools)  # 工具循环非流式,无逐 token
        return adapter.respond(ctx, on_token=on_token)

    # ── §3.2 Context 压缩层 ────────────────────────────────────────────
    def summarize_round(self, msgs: List[InternalMessage]) -> List[RoundSummary]:
        """每轮结束压成结构化摘要（利用发言末尾自带的结构化字段）。"""
        return [m.parse_structured() for m in msgs]

    def build_next_context(
        self,
        user_q: InternalMessage,
        last_summaries: List[RoundSummary],
    ) -> List[InternalMessage]:
        """传 [用户原始问题] + [上一轮各方 RoundSummary]，而非原始全文（§3.2）。"""
        ctx: List[InternalMessage] = [user_q]
        for s in last_summaries:
            ctx.append(
                InternalMessage(
                    role="agent",
                    agent_id=s.agent_id,
                    content=s.render(),
                    refs=list(s.rebuts),
                    round=s.round,
                )
            )
        return ctx

    # ── §3.3 双闸门 ────────────────────────────────────────────────────
    def turn_limit_reached(self, n: int) -> bool:
        """Turn-Counter：防死循环 / 礼貌性附和拉锯（§3.3）。"""
        return n >= self.n_max

    def _charge_round(self, msgs: List[InternalMessage]) -> None:
        cost = sum(_estimate_tokens(m) for m in msgs)
        self.ledger.charge(cost)
        # Global token budget：整场会议成本熔断（§3.3 / §10.4）。
        if self.ledger.total > self.global_token_budget:
            raise BudgetExceeded(
                f"全局 token 预算 {self.global_token_budget} 已被击穿："
                f"{self.ledger.total}"
            )
