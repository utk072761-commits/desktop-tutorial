"""闭合状态机（蓝图 §1）。

v0 的状态机不闭合：PROPOSAL 被拒后无去向、COMMITTED 之后无回路、
用户中途介入无对应转换。此处按蓝图把转换表显式枚举并校验，
非法转换直接抛错——状态机是整套系统的物理护栏（§7.3 / §10.5）。
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, Set, Tuple


class State(str, Enum):
    WAITING = "WAITING"                      # 等待用户指令
    DISPATCHING = "DISPATCHING"              # 解析 @ 标签，选定激活子集 S，并发分发
    DISCUSSION = "DISCUSSION"                # 辩论进行中（仅 DebateMode）
    CONVERGENCE_CHECK = "CONVERGENCE_CHECK"  # 独立裁判判定是否收敛
    PROPOSAL = "PROPOSAL"                    # 分歧矩阵已生成，呈递 Decision Board
    COMMITTED = "COMMITTED"                  # 用户通过，执行外部操作
    REJECTED = "REJECTED"                    # 用户拒绝（瞬时态，立即转出）


class Transition(str, Enum):
    USER_INPUT = "user_input"
    SINGLE_MODE = "single_mode"
    DEBATE_MODE = "debate_mode"
    ROUND_END = "round_end"
    NOT_CONVERGED = "not_converged"          # 且 n < n_max
    CONVERGED = "converged"                  # 或 n >= n_max
    USER_INTERRUPT = "user_interrupt"        # 辩论中强制截断进 PROPOSAL
    USER_APPROVE = "user_approve"
    USER_REJECT = "user_reject"
    WITH_NEW_INSTRUCTION = "with_new_instruction"
    REOPEN_DEBATE = "reopen_debate"
    DONE = "done"
    USER_ABORT = "user_abort"                # 全局中断，任何状态回 WAITING


# 显式转换表：(from_state, transition) -> to_state（§1）。
_TABLE: Dict[Tuple[State, Transition], State] = {
    (State.WAITING, Transition.USER_INPUT): State.DISPATCHING,
    (State.DISPATCHING, Transition.SINGLE_MODE): State.WAITING,
    (State.DISPATCHING, Transition.DEBATE_MODE): State.DISCUSSION,
    (State.DISCUSSION, Transition.ROUND_END): State.CONVERGENCE_CHECK,
    (State.CONVERGENCE_CHECK, Transition.NOT_CONVERGED): State.DISCUSSION,
    (State.CONVERGENCE_CHECK, Transition.CONVERGED): State.PROPOSAL,
    (State.DISCUSSION, Transition.USER_INTERRUPT): State.PROPOSAL,
    (State.PROPOSAL, Transition.USER_APPROVE): State.COMMITTED,
    (State.PROPOSAL, Transition.USER_REJECT): State.REJECTED,
    (State.REJECTED, Transition.WITH_NEW_INSTRUCTION): State.DISPATCHING,
    (State.REJECTED, Transition.REOPEN_DEBATE): State.DISCUSSION,
    (State.COMMITTED, Transition.DONE): State.WAITING,
}

# 全局边：任何状态 --user_abort--> WAITING（§1 要点）。
_GLOBAL_EDGES: Dict[Transition, State] = {
    Transition.USER_ABORT: State.WAITING,
}


class IllegalTransition(Exception):
    """非法状态转换。"""


class StateMachine:
    """轻量状态机，强制只能走转换表里声明的边。"""

    def __init__(self, state: State = State.WAITING) -> None:
        self.state = state
        self.history: list[Tuple[State, Transition, State]] = []

    def can(self, t: Transition) -> bool:
        return t in _GLOBAL_EDGES or (self.state, t) in _TABLE

    def fire(self, t: Transition) -> State:
        """执行一次转换，返回新状态；非法则抛 IllegalTransition。"""
        if t in _GLOBAL_EDGES:
            target = _GLOBAL_EDGES[t]
        elif (self.state, t) in _TABLE:
            target = _TABLE[(self.state, t)]
        else:
            raise IllegalTransition(
                f"状态 {self.state.value} 不接受转换 {t.value}"
            )
        self.history.append((self.state, t, target))
        self.state = target
        return self.state
