"""骨架边界扩展单测：真实 adapter 契约、持久化、LLM 裁判/压缩、UI 后端。

全部离线：provider adapter 只测纯逻辑（payload 构造 / 分帧解析），不发网络；
LLM 裁判/压缩用 fake completer 注入；UI 直接驱动 RoundtableServer 对象。
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from roundtable import (
    ClaudeAdapter,
    GeminiAdapter,
    GrokAdapter,
    InternalMessage,
    LLMCompressor,
    LLMJudge,
    MockAdapter,
    Roundtable,
    SessionStore,
    session_from_dict,
    session_to_dict,
)
from roundtable.providers import _iter_sse
from roundtable.state import State
from roundtable.ui.server import RoundtableServer


def run(coro):
    return asyncio.run(coro)


# ── §2.2 / §10.1 真实 adapter 契约 ──────────────────────────────────────
def _ctx():
    return [
        InternalMessage(role="user", agent_id="user", content="单体还是微服务？", round=0),
        InternalMessage(role="agent", agent_id="A1", content="我主张单体", round=1),
        InternalMessage(role="agent", agent_id="A2", content="我主张微服务", round=1),
    ]


def test_adapters_have_version_markers():
    # §10.1：每个 adapter 必须带版本标记。
    assert ClaudeAdapter("A1").api_version
    assert GeminiAdapter("A1").api_version
    assert GrokAdapter("A1").api_version


def test_claude_to_native_role_mapping():
    payload = ClaudeAdapter("A1").to_native(_ctx())
    assert payload["model"] == "claude-opus-4-8"
    assert payload["system"]
    roles = [m["role"] for m in payload["messages"]]
    assert roles[0] == "user"               # 首条必须 user
    # 本方 A1 -> assistant，他方 -> user
    a1 = [m for m in payload["messages"] if m["content"].startswith("我主张单体")]
    assert a1 and a1[0]["role"] == "assistant"
    a2 = [m for m in payload["messages"] if "[A2]" in m["content"]]
    assert a2 and a2[0]["role"] == "user"
    # Opus 4.8 不可传 temperature/budget_tokens。
    assert "temperature" not in payload and "thinking" not in payload


def test_gemini_to_native_and_extract():
    payload = GeminiAdapter("A1").to_native(_ctx())
    assert payload["system_instruction"]["parts"][0]["text"]
    assert payload["contents"][0]["role"] == "user"
    self_part = [c for c in payload["contents"] if c["parts"][0]["text"].startswith("我主张单体")]
    assert self_part and self_part[0]["role"] == "model"
    chunk = {"candidates": [{"content": {"parts": [{"text": "你好"}, {"text": "世界"}]}}]}
    assert GeminiAdapter._extract_text(chunk) == "你好世界"


def test_grok_to_native_and_extract():
    payload = GrokAdapter("A1").to_native(_ctx())
    assert payload["messages"][0]["role"] == "system"
    assert payload["stream"] is True
    chunk = {"choices": [{"delta": {"content": "片段"}}]}
    assert GrokAdapter._extract_text(chunk) == "片段"
    assert GrokAdapter._extract_text({"choices": [{"delta": {}}]}) == ""


def test_from_native_parses_refs():
    resp = {
        "payload": {"_round": 2},
        "text": "反对。\nSTANCE: 单体\nARGS: a; b\nREBUTS: A2",
    }
    msg = ClaudeAdapter("A1").from_native(resp)
    assert msg.agent_id == "A1" and msg.round == 2 and msg.refs == ["A2"]


def test_iter_sse_parsing():
    async def lines():
        for ln in [": comment", "", "data: {\"x\":1}", "event: ping", "data: [DONE]"]:
            yield ln

    async def collect():
        return [d async for d in _iter_sse(lines())]

    assert run(collect()) == ['{"x":1}', "[DONE]"]


# ── §8 持久化 ────────────────────────────────────────────────────────────
def _debated_rt():
    reg = {
        "A1": MockAdapter("A1", stance_seed="单体"),
        "A2": MockAdapter("A2", stance_seed="微服务", concede_at=1, concede_to="A1"),
    }
    rt = Roundtable("persist-1", reg)
    run(rt.run_debate("辩", n_max=3))
    return rt


def test_session_dict_round_trip():
    rt = _debated_rt()
    d = session_to_dict(rt.session)
    import json
    json.dumps(d, ensure_ascii=False)  # 必须可 JSON 序列化
    s2 = session_from_dict(d)
    assert s2.session_id == rt.session.session_id
    assert s2.state == rt.session.state
    assert len(s2.messages) == len(rt.session.messages)
    assert s2.dispute_matrix is not None
    assert s2.verdict.converged == rt.session.verdict.converged


def test_sqlite_store(tmp_path):
    rt = _debated_rt()
    store = SessionStore(str(tmp_path / "rt.db"))
    store.save(rt.session)
    assert "persist-1" in store.list_sessions()
    loaded = store.load("persist-1")
    assert loaded.state == State.PROPOSAL
    assert len(loaded.summaries) == len(rt.session.summaries)
    store.delete("persist-1")
    assert store.load("persist-1") is None
    store.close()


# ── §4 / §3.2 LLM 裁判与压缩（fake completer，离线）─────────────────────
def test_llm_judge_parses_verdict():
    async def fake(system, user):
        return '{"converged": true, "agreement_points": ["统一用单体"], ' \
               '"open_disputes": [], "confidence": 0.9}'

    judge = LLMJudge(fake)
    summaries = MockAdapter("A1").from_native(
        {"payload": {"round": 1}, "text": "STANCE: 单体\nARGS: a\nREBUTS:"}
    ).parse_structured()
    v = run(judge.verdict([summaries]))
    assert v.converged and v.confidence == 0.9 and v.agreement_points == ["统一用单体"]


def test_llm_compressor_parses_summary():
    async def fake(system, user):
        return '{"stance": "单体", "key_arguments": ["运维省", "边界稳"], "rebuts": ["A2"]}'

    rs = run(LLMCompressor(fake).summarize("A1", "正文……", round=1))
    assert rs.stance == "单体" and rs.key_arguments == ["运维省", "边界稳"] and rs.rebuts == ["A2"]


def test_roundtable_with_async_llm_judge():
    # Roundtable 必须能驱动异步裁判（LLMJudge），并据其判定收敛。
    async def fake(system, user):
        return '{"converged": true, "agreement_points": ["一致"], "open_disputes": [], "confidence": 1.0}'

    reg = {"A1": MockAdapter("A1", stance_seed="单体"), "A2": MockAdapter("A2", stance_seed="微服务")}
    rt = Roundtable("llm-1", reg, judge=LLMJudge(fake))
    run(rt.run_debate("辩", n_max=5))
    assert rt.session.verdict.converged
    assert rt.session.round_counter == 1  # 裁判第 1 轮即判收敛
    assert rt.session.state == State.PROPOSAL


# ── 流式回调 + 真实中断探针 ─────────────────────────────────────────────
def test_on_token_streams_per_agent():
    reg = {"A1": MockAdapter("A1", stance_seed="单体"), "A2": MockAdapter("A2", stance_seed="微服务")}
    from roundtable import Moderator

    received = {}

    def on_token(aid, tok):
        received[aid] = received.get(aid, "") + tok

    mod = Moderator(reg)
    user = InternalMessage(role="user", agent_id="user", content="辩", round=0)
    msgs = run(mod.run_round({"A1", "A2"}, [user], on_token=on_token))
    assert set(received) == {"A1", "A2"}
    # 回调累积的内容与终稿一致（终稿做了 strip）。
    for m in msgs:
        assert received[m.agent_id].strip() == m.content


def test_interrupt_probe_truncates_to_proposal():
    from roundtable import Judge as _J, Roundtable as _R

    reg = {"A1": MockAdapter("A1", stance_seed="单体"), "A2": MockAdapter("A2", stance_seed="微服务")}
    rt = _R("probe-1", reg, judge=_J(agreement_threshold=0.99))  # 永不收敛
    rounds_seen = []
    run(rt.run_debate("辩到底", n_max=5,
                      on_round=rounds_seen.append,
                      interrupt=lambda: True))  # 第 1 轮结束即中断
    assert rt.session.round_counter == 1
    assert rounds_seen == [1]
    assert rt.session.state == State.PROPOSAL


# ── §7 UI 后端 ───────────────────────────────────────────────────────────
def test_ui_server_full_flow():
    app = RoundtableServer()
    payload = app.start("架构选型：单体 vs 微服务？", debate=True, n_max=5)
    assert payload["state"] == "PROPOSAL"
    assert payload["session"]["dispute_matrix"] is not None
    # 通过前不得有副作用记录。
    assert payload["committed"] == []
    out = app.approve()
    assert out["state"] == "WAITING"
    assert out["committed"]  # 副作用在 COMMITTED 之后执行


def test_ui_server_surfaces_tool_trace():
    app = RoundtableServer()
    payload = app.start("微服务成本多高？", debate=False, n_max=5, use_tools=True)
    msgs = payload["session"]["messages"]
    a1 = next(m for m in msgs if m["agent_id"] == "A1")
    assert a1["tool_trace"], "tool_trace 应出现在 UI 状态里供圆桌视窗渲染"
    assert a1["tool_trace"][0]["name"] == "search"
    assert "基准数据" in a1["tool_trace"][0]["result"]


def test_tool_trace_survives_serialization():
    from roundtable import MockAdapter, Roundtable
    from roundtable.ui.server import demo_tools

    reg = {"A1": MockAdapter("A1", tool_call="search", tool_args={"q": "x"})}
    rt = Roundtable("trace-1", reg)
    run(rt.ask_single("查", tools=demo_tools()))
    s2 = session_from_dict(session_to_dict(rt.session))
    a1 = next(m for m in s2.messages if m.agent_id == "A1")
    assert a1.tool_trace and a1.tool_trace[0]["name"] == "search"


def test_ui_server_single_mode():
    app = RoundtableServer()
    payload = app.start("@A1 给个方案", debate=False, n_max=5)
    assert payload["state"] == "WAITING"
    assert payload["session"]["messages"]


def test_ui_server_reject_reopens():
    app = RoundtableServer()
    app.start("辩", debate=True, n_max=5)
    out = app.reject(reopen=True, new_instruction=False)
    assert out["state"] == "DISCUSSION"


def test_ui_server_background_with_live_stream():
    import time

    app = RoundtableServer()
    d = app.start("辩", debate=True, n_max=5, background=True, latency=0.3)
    assert d["busy"] is True  # 立即返回,后台进行中

    saw_live = False
    deadline = time.time() + 10
    while time.time() < deadline:
        d = app.state_payload()
        if d["live"]:
            saw_live = True
        if not d["busy"]:
            break
        time.sleep(0.05)
    assert saw_live, "辩论进行中应能观察到逐 token 流式缓冲"
    assert d["busy"] is False and d["live"] == {}
    assert d["state"] == "PROPOSAL"


def test_ui_server_interrupt_endpoint():
    import time

    app = RoundtableServer()
    app.start("辩", debate=True, n_max=5, background=True, latency=0.3)
    app.interrupt()  # 第 1 轮结束后应截断进 PROPOSAL

    deadline = time.time() + 10
    while time.time() < deadline and app.busy:
        time.sleep(0.05)
    d = app.state_payload()
    assert d["state"] == "PROPOSAL"
    assert d["session"]["round_counter"] == 1  # 没辩满 5 轮就被介入截断


def test_ui_server_abort_while_busy_routes_to_interrupt():
    import time

    app = RoundtableServer()
    app.start("辩", debate=True, n_max=5, background=True, latency=0.3)
    app.abort()  # busy 时 abort 走中断探针,轮边界优雅截断

    deadline = time.time() + 10
    while time.time() < deadline and app.busy:
        time.sleep(0.05)
    assert app.state_payload()["state"] == "PROPOSAL"


if __name__ == "__main__":
    import subprocess
    raise SystemExit(subprocess.call(["pytest", "-q", __file__]))
