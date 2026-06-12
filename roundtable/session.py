"""会话状态存储（§8）+ 顶层编排器（贯穿 §1 状态机的端到端流程）。

Roundtable 把 dispatcher / moderator / judge / decision 串成一条
受状态机约束的流水线，并守住唯一的物理红线：
**PROPOSAL → COMMITTED 是唯一能触发外部操作的边**（§7.3 / §10.5）。
用户按下"通过"前，系统不得执行任何有副作用的动作。
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Set

if TYPE_CHECKING:
    from .tools import Tool

from .adapters import AgentAdapter
from .decision import build_dispute_matrix
from .dispatcher import resolve_active_set
from .judge import Judge
from .messages import (
    ConvergenceVerdict,
    DisputeMatrix,
    InternalMessage,
    RoundSummary,
)
from .moderator import BudgetExceeded, Moderator
from .state import IllegalTransition, State, StateMachine, Transition


# 协作协议（注入各模型 System Prompt，§6）。
SYSTEM_PROMPT = """\
你正在参与一个多智能体圆桌会议,主持人(用户)拥有最终裁决权。

1. 单问模式:用户未开启辩论时,只回答用户的问题,不提及其他参会者。
2. 辩论模式:
   - 你必须明确给出你的【立场】,并附 2-3 条【核心论据】。
   - 质疑他方时,必须指明【具体差异点】(@对方 + 哪一条论据),不得泛泛附和或泛泛反对。
   - 不要宣布"我们达成一致"——是否收敛由裁判判定,不归你说。
   - 若你确实改变立场,明说"我接受 @X 关于 Y 的论据",并更新你的立场。
3. 输出末尾附结构化字段(供系统压缩 context):
   STANCE: <一句话>
   ARGS: <2-3 条>
   REBUTS: <你反驳的 agent_id,无则空>
4. 用户随时可中断。收到中断信号立即停止,等待裁决。
"""


@dataclass
class Session:
    """会话状态（§8）。messages 全量留存做审计/回放；喂模型用 summaries。"""

    session_id: str
    state: State = State.WAITING
    active_set: Set[str] = field(default_factory=set)
    debate_mode: bool = False
    round_counter: int = 0
    messages: List[InternalMessage] = field(default_factory=list)       # 全量发言
    summaries: List[RoundSummary] = field(default_factory=list)         # 压缩后的轮摘要
    verdict: Optional[ConvergenceVerdict] = None                        # 最近裁决
    dispute_matrix: Optional[DisputeMatrix] = None                      # 当前决策板
    token_ledger: Dict[str, object] = field(default_factory=dict)       # 成本账本快照


class Roundtable:
    """顶层编排器：受状态机约束地推进一次会议。"""

    def __init__(
        self,
        session_id: str,
        registry: Dict[str, AgentAdapter],
        judge: Optional[Judge] = None,
        moderator: Optional[Moderator] = None,
    ) -> None:
        self.session = Session(session_id=session_id)
        self.registry = registry
        self.judge = judge or Judge()
        self.moderator = moderator or Moderator(registry)
        self.sm = StateMachine(self.session.state)

    # ── 状态机驱动小工具 ───────────────────────────────────────────────
    def _fire(self, t: Transition) -> None:
        self.sm.fire(t)
        self.session.state = self.sm.state

    def abort(self) -> None:
        """全局中断：任何状态回 WAITING（§1）。"""
        self._fire(Transition.USER_ABORT)

    async def _verdict(self, summaries: List[RoundSummary]) -> ConvergenceVerdict:
        """调用裁判；兼容同步（heuristic Judge）与异步（LLMJudge）两种实现。"""
        result = self.judge.verdict(summaries)
        if inspect.isawaitable(result):
            result = await result
        return result

    # ── 单问模式（DISPATCHING --single_mode--> WAITING）────────────────
    async def ask_single(
        self, prompt: str, tools: Optional[List["Tool"]] = None, on_token=None
    ) -> List[InternalMessage]:
        """非辩论：直接展示答案，不进辩论流。可带 tools 让模型先调工具再作答。"""
        user_msg = InternalMessage(role="user", agent_id="user", content=prompt, round=0)
        self.session.messages.append(user_msg)
        self._fire(Transition.USER_INPUT)  # WAITING -> DISPATCHING

        self.session.debate_mode = False
        active = resolve_active_set(prompt, self.registry)
        self.session.active_set = active

        answers = await self.moderator.run_round(
            active, [user_msg], tools=tools, on_token=on_token
        )
        self.session.messages.extend(answers)
        self._fire(Transition.SINGLE_MODE)  # DISPATCHING -> WAITING
        return answers

    # ── 辩论模式（DISPATCHING --debate_mode--> DISCUSSION ... PROPOSAL）─
    async def run_debate(
        self,
        prompt: str,
        n_max: Optional[int] = None,
        user_interrupt_after: Optional[int] = None,
        tools: Optional[List["Tool"]] = None,
        on_token=None,
        on_round: Optional[Callable[[int], None]] = None,
        interrupt: Optional[Callable[[], bool]] = None,
    ) -> DisputeMatrix:
        """跑完整辩论直到 PROPOSAL，返回分歧矩阵（决策板）。

        user_interrupt_after: 测试/演示用——在第 k 轮后模拟用户中断，
        触发 DISCUSSION --user_interrupt--> PROPOSAL（强制截断）。
        interrupt: 真实用户中断探针——每轮结束后查询，返回 True 即走同一条
        user_interrupt 边（UI「介入辩论」按钮经此接入，§1/§7.3）。
        on_token(agent_id, token): 逐 token 回调，供 UI 实时渲染。
        on_round(n): 每轮发言落定后回调（UI 据此清空流式缓冲）。
        """
        if n_max is not None:
            self.moderator.n_max = n_max

        user_msg = InternalMessage(role="user", agent_id="user", content=prompt, round=0)
        self.session.messages.append(user_msg)
        self._fire(Transition.USER_INPUT)  # WAITING -> DISPATCHING

        self.session.debate_mode = True
        active = resolve_active_set(prompt, self.registry)
        self.session.active_set = active
        self._fire(Transition.DEBATE_MODE)  # DISPATCHING -> DISCUSSION

        ctx: List[InternalMessage] = [user_msg]
        last_summaries: List[RoundSummary] = []

        while True:
            self.session.round_counter += 1
            n = self.session.round_counter

            try:
                msgs = await self.moderator.run_round(
                    active, ctx, tools=tools, on_token=on_token
                )
            except BudgetExceeded:
                # 全局熔断：直接截断进 PROPOSAL（§3.3）。
                break

            self.session.messages.extend(msgs)
            last_summaries = self.moderator.summarize_round(msgs)
            self.session.summaries.extend(last_summaries)
            if on_round is not None:
                on_round(n)

            # 用户中断强制截断当前进度进 PROPOSAL（§1）：
            # 模拟参数（测试用）与真实中断探针（UI 按钮）走同一条边。
            if (user_interrupt_after is not None and n >= user_interrupt_after) or (
                interrupt is not None and interrupt()
            ):
                self._fire(Transition.USER_INTERRUPT)  # DISCUSSION -> PROPOSAL
                break

            # DISCUSSION --round_end--> CONVERGENCE_CHECK
            self._fire(Transition.ROUND_END)

            # §4 外置裁判判定（不在 DISCUSSION 内部自评）。
            verdict = await self._verdict(last_summaries)
            self.session.verdict = verdict

            if verdict.converged or self.moderator.turn_limit_reached(n):
                self._fire(Transition.CONVERGED)  # -> PROPOSAL
                break

            self._fire(Transition.NOT_CONVERGED)  # -> DISCUSSION（继续下一轮）
            ctx = self.moderator.build_next_context(user_msg, last_summaries)

        # 若因中断/熔断未走裁判，则补一次裁决供分歧矩阵使用。
        if self.session.verdict is None:
            self.session.verdict = await self._verdict(last_summaries)

        self.session.dispute_matrix = build_dispute_matrix(
            self.session.verdict, last_summaries
        )
        self.session.token_ledger = {
            "per_round": list(self.moderator.ledger.per_round),
            "total": self.moderator.ledger.total,
        }
        assert self.session.state == State.PROPOSAL
        return self.session.dispute_matrix

    # ── 终审闭环（§7.3：副作用只挂在 COMMITTED 之后）──────────────────
    def approve(self, side_effect: Optional[Callable[[], object]] = None) -> object:
        """PROPOSAL --user_approve--> COMMITTED，执行外部操作后回 WAITING。

        side_effect 是唯一被允许产生副作用的入口；它只能在此处、
        即状态已合法跨过 PROPOSAL→COMMITTED 后被调用。其它任何状态
        下尝试执行副作用都应被状态机拒绝。
        """
        if self.session.state != State.PROPOSAL:
            raise IllegalTransition("只有 PROPOSAL 状态可被通过")
        self._fire(Transition.USER_APPROVE)  # -> COMMITTED

        result = side_effect() if side_effect else None

        self._fire(Transition.DONE)  # COMMITTED -> WAITING
        return result

    def reject(self, reopen: bool = False, new_instruction: bool = False) -> State:
        """PROPOSAL --user_reject--> REJECTED，再按用户意图转出（§1）。"""
        if self.session.state != State.PROPOSAL:
            raise IllegalTransition("只有 PROPOSAL 状态可被拒绝")
        self._fire(Transition.USER_REJECT)  # -> REJECTED（瞬时态）
        if new_instruction:
            self._fire(Transition.WITH_NEW_INSTRUCTION)  # -> DISPATCHING
        elif reopen:
            self._fire(Transition.REOPEN_DEBATE)  # -> DISCUSSION
        return self.session.state
