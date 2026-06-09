"""圆桌会议骨架单测（不依赖网络）。

覆盖蓝图关键不变量：
  - §1 状态机闭合 + 非法转换被拒 + 全局中断
  - §2.3 点名/激活子集
  - §2.1/§6 结构化字段解析 -> RoundSummary
  - §3.1 并发 + 失败隔离 + 固定座次
  - §3.2 context 压缩（喂摘要而非全文）
  - §3.3 turn-counter 强制收敛 / 全局 token 熔断
  - §4 外置裁判收敛判定
  - §5 分歧矩阵（非共识摘要）
  - §7.3 副作用只挂在 COMMITTED 之后
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from roundtable import (
    InternalMessage,
    Judge,
    MockAdapter,
    Moderator,
    Roundtable,
    parse_mentions,
    resolve_active_set,
)
from roundtable.decision import build_dispute_matrix
from roundtable.moderator import BudgetExceeded
from roundtable.state import (
    IllegalTransition,
    State,
    StateMachine,
    Transition,
)


def run(coro):
    return asyncio.run(coro)


def _registry(**kwargs):
    return {
        "A1": MockAdapter("A1", stance_seed="单体", **kwargs.get("A1", {})),
        "A2": MockAdapter("A2", stance_seed="微服务", **kwargs.get("A2", {})),
        "A3": MockAdapter("A3", stance_seed="折中", **kwargs.get("A3", {})),
    }


# ── §1 状态机 ────────────────────────────────────────────────────────────
def test_state_machine_happy_path():
    sm = StateMachine()
    assert sm.state == State.WAITING
    sm.fire(Transition.USER_INPUT)
    assert sm.state == State.DISPATCHING
    sm.fire(Transition.DEBATE_MODE)
    sm.fire(Transition.ROUND_END)
    sm.fire(Transition.CONVERGED)
    assert sm.state == State.PROPOSAL
    sm.fire(Transition.USER_APPROVE)
    assert sm.state == State.COMMITTED
    sm.fire(Transition.DONE)
    assert sm.state == State.WAITING


def test_illegal_transition_rejected():
    sm = StateMachine()
    with pytest.raises(IllegalTransition):
        sm.fire(Transition.USER_APPROVE)  # WAITING 不能直接 approve


def test_global_abort_from_any_state():
    sm = StateMachine()
    sm.fire(Transition.USER_INPUT)
    sm.fire(Transition.DEBATE_MODE)
    assert sm.state == State.DISCUSSION
    sm.fire(Transition.USER_ABORT)  # 全局边
    assert sm.state == State.WAITING


def test_rejected_reopen_and_new_instruction():
    sm = StateMachine(State.PROPOSAL)
    sm.fire(Transition.USER_REJECT)
    assert sm.state == State.REJECTED
    sm.fire(Transition.REOPEN_DEBATE)
    assert sm.state == State.DISCUSSION


# ── §2.3 点名 ────────────────────────────────────────────────────────────
def test_parse_mentions():
    assert parse_mentions("@A1 和 @A2 你们看") == {"A1", "A2"}
    assert parse_mentions("没有点名") == set()


def test_resolve_active_set():
    reg = _registry()
    assert resolve_active_set("@A1 @A2 上", reg) == {"A1", "A2"}
    assert resolve_active_set("全员", reg) == {"A1", "A2", "A3"}
    # 点了未注册的成员 -> 忽略, 回落到该交集（此处为空 -> 全员）。
    assert resolve_active_set("@ZZ", reg) == {"A1", "A2", "A3"}


# ── §2.1/§6 结构化解析 ──────────────────────────────────────────────────
def test_parse_structured_fields():
    msg = InternalMessage(
        role="agent",
        agent_id="A1",
        content="我反对。@A2 你的第1条不成立。\n"
        "STANCE: 应采用单体\nARGS: 运维成本低; 边界易变\nREBUTS: A2",
        round=1,
    )
    s = msg.parse_structured()
    assert s.stance == "应采用单体"
    assert s.key_arguments == ["运维成本低", "边界易变"]
    assert s.rebuts == ["A2"]


def test_parse_structured_degrades_without_fields():
    msg = InternalMessage(role="agent", agent_id="A1", content="只有一句话没有字段", round=1)
    s = msg.parse_structured()
    assert s.stance == "只有一句话没有字段"


# ── §3.1 并发 + 失败隔离 + 座次 ─────────────────────────────────────────
def test_run_round_concurrent_and_ordered():
    reg = _registry()
    mod = Moderator(reg)
    user = InternalMessage(role="user", agent_id="user", content="辩", round=0)
    msgs = run(mod.run_round({"A1", "A2", "A3"}, [user]))
    assert [m.agent_id for m in msgs] == ["A1", "A2", "A3"]  # 固定座次


def test_run_round_failure_isolation():
    reg = _registry()
    reg["A2"] = MockAdapter("A2", stance_seed="微服务", fail=True)
    mod = Moderator(reg)
    user = InternalMessage(role="user", agent_id="user", content="辩", round=0)
    msgs = run(mod.run_round({"A1", "A2", "A3"}, [user]))
    assert [m.agent_id for m in msgs] == ["A1", "A3"]  # A2 降级为空缺


# ── §3.2 context 压缩 ───────────────────────────────────────────────────
def test_build_next_context_uses_summaries_not_fulltext():
    reg = _registry()
    mod = Moderator(reg)
    user = InternalMessage(role="user", agent_id="user", content="原始问题", round=0)
    msgs = run(mod.run_round({"A1", "A2"}, [user]))
    summaries = mod.summarize_round(msgs)
    ctx = mod.build_next_context(user, summaries)
    # 下一轮 context = 用户原始问题 + 各方摘要，且摘要比原文短。
    assert ctx[0].content == "原始问题"
    assert len(ctx) == 1 + len(summaries)
    assert all(len(c.content) <= len(m.content) + 80 for c, m in zip(ctx[1:], msgs))


# ── §3.3 闸门 ────────────────────────────────────────────────────────────
def test_turn_counter_forces_proposal():
    # 三方永不让步 -> 永不收敛 -> 必须靠 n_max 强制进 PROPOSAL。
    reg = _registry()
    rt = Roundtable("t", reg, judge=Judge(agreement_threshold=0.99))
    matrix = run(rt.run_debate("辩到底", n_max=3))
    assert rt.session.round_counter == 3
    assert rt.session.state == State.PROPOSAL
    assert matrix.disputes  # 未收敛 -> 有分歧


def test_global_token_budget_circuit_breaker():
    reg = _registry()
    mod = Moderator(reg, global_token_budget=1)  # 极小预算 -> 立即熔断
    user = InternalMessage(role="user", agent_id="user", content="辩", round=0)
    with pytest.raises(BudgetExceeded):
        run(mod.run_round({"A1", "A2", "A3"}, [user]))


# ── §4 外置裁判 ──────────────────────────────────────────────────────────
def test_judge_detects_convergence():
    reg = _registry(
        A2={"concede_at": 1, "concede_to": "A1"},
        A3={"concede_at": 1, "concede_to": "A1"},
    )
    rt = Roundtable("t", reg)
    run(rt.run_debate("收敛吧", n_max=5))
    assert rt.session.verdict.converged
    assert rt.session.round_counter < 5  # 提前收敛, 没耗到 n_max


# ── §5 分歧矩阵 ──────────────────────────────────────────────────────────
def test_dispute_matrix_shows_strongest_case():
    reg = _registry()
    rt = Roundtable("t", reg, judge=Judge(agreement_threshold=0.99))
    matrix = run(rt.run_debate("辩", n_max=2))
    assert matrix.disputes
    d = matrix.disputes[0]
    assert set(d.positions) <= {"A1", "A2", "A3"}
    # 每个有立场的一方都附最强论据原话。
    for aid in d.positions:
        assert d.strongest_case.get(aid)


# ── §7.3 副作用边界 ─────────────────────────────────────────────────────
def test_side_effect_only_after_committed():
    reg = _registry()
    rt = Roundtable("t", reg, judge=Judge(agreement_threshold=0.99))

    fired = {"count": 0}

    def effect():
        fired["count"] += 1
        return "done"

    # PROPOSAL 之前不可能调用 approve（状态非 PROPOSAL）。
    with pytest.raises(IllegalTransition):
        rt.approve(side_effect=effect)
    assert fired["count"] == 0

    run(rt.run_debate("辩", n_max=2))
    assert rt.session.state == State.PROPOSAL
    res = rt.approve(side_effect=effect)
    assert res == "done"
    assert fired["count"] == 1
    assert rt.session.state == State.WAITING


def test_user_interrupt_truncates_to_proposal():
    reg = _registry()
    rt = Roundtable("t", reg, judge=Judge(agreement_threshold=0.99))
    run(rt.run_debate("辩", n_max=5, user_interrupt_after=1))
    assert rt.session.round_counter == 1
    assert rt.session.state == State.PROPOSAL


if __name__ == "__main__":
    import subprocess

    raise SystemExit(subprocess.call(["pytest", "-q", __file__]))
