"""tool_use 单测:统一工具循环 + 三家 provider 的工具协议(纯逻辑,离线)。

provider 钩子用合成响应 dict 验证;端到端工具循环用 MockAdapter 驱动,
工具 handler 带可观测副作用(计数器),确认确实被调用、结果回灌进终稿。
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
    MockAdapter,
    Moderator,
    Roundtable,
    Tool,
    ToolCall,
    ToolResult,
    run_tool_loop,
)
from roundtable.state import State
from roundtable.tools import execute


def run(coro):
    return asyncio.run(coro)


def _search_tool(counter):
    def handler(args):
        counter["n"] += 1
        counter["last_args"] = args
        return f"命中:{args.get('q', '')}"

    return Tool(
        name="search",
        description="检索资料",
        input_schema={
            "type": "object",
            "properties": {"q": {"type": "string"}},
            "required": ["q"],
        },
        handler=handler,
    )


def _ctx():
    return [InternalMessage(role="user", agent_id="user", content="该用什么架构？", round=0)]


# ── 统一类型 + execute ──────────────────────────────────────────────────
def test_execute_runs_handler():
    counter = {"n": 0}
    tool = _search_tool(counter)
    res = run(execute(tool, ToolCall("c1", "search", {"q": "微服务"})))
    assert res.content == "命中:微服务" and not res.is_error and counter["n"] == 1


def test_execute_isolates_exception():
    def boom(args):
        raise ValueError("炸了")

    tool = Tool("x", "d", {"type": "object"}, boom)
    res = run(execute(tool, ToolCall("c", "x", {})))
    assert res.is_error and "炸了" in res.content


async def _async_handler(args):
    return "async-ok"


def test_execute_supports_async_handler():
    tool = Tool("a", "d", {"type": "object"}, _async_handler)
    res = run(execute(tool, ToolCall("c", "a", {})))
    assert res.content == "async-ok"


# ── 离线端到端工具循环（MockAdapter）─────────────────────────────────────
def test_mock_tool_loop_end_to_end():
    counter = {"n": 0}
    tool = _search_tool(counter)
    mock = MockAdapter("A1", stance_seed="先查再答", tool_call="search", tool_args={"q": "架构"})
    msg = run(run_tool_loop(mock, _ctx(), [tool]))
    assert counter["n"] == 1                       # 工具确实被调用一次
    assert counter["last_args"] == {"q": "架构"}
    assert "已查询工具" in msg.content             # 结果回灌进终稿
    assert "命中:架构" in msg.content
    assert msg.agent_id == "A1"


def test_tool_loop_populates_trace():
    counter = {"n": 0}
    tool = _search_tool(counter)
    mock = MockAdapter("A1", tool_call="search", tool_args={"q": "架构"})
    msg = run(run_tool_loop(mock, _ctx(), [tool]))
    assert len(msg.tool_trace) == 1
    entry = msg.tool_trace[0]
    assert entry["name"] == "search"
    assert entry["arguments"] == {"q": "架构"}
    assert entry["result"] == "命中:架构" and entry["is_error"] is False


def test_tool_loop_trace_records_unknown_tool_error():
    mock = MockAdapter("A1", tool_call="ghost", tool_args={})
    msg = run(run_tool_loop(mock, _ctx(), []))
    assert msg.tool_trace and msg.tool_trace[0]["is_error"] is True


def test_tool_loop_without_tools_is_plain_answer():
    counter = {"n": 0}
    # 未设 tool_call -> 即便给了工具也不调用,直接终稿。
    mock = MockAdapter("A1", stance_seed="直接答")
    msg = run(run_tool_loop(mock, _ctx(), [_search_tool(counter)]))
    assert counter["n"] == 0 and "STANCE" in msg.content


def test_tool_loop_unknown_tool_call_is_isolated():
    # 模型点了一个未注册的工具 -> 记为错误结果,不抛、不死循环。
    mock = MockAdapter("A1", tool_call="ghost", tool_args={})
    msg = run(run_tool_loop(mock, _ctx(), []))  # 工具表为空 -> ghost 未知
    assert "STANCE" in msg.content  # 仍能收口到终稿


def test_unsupported_adapter_rejected():
    class Bare(MockAdapter):
        supports_tools = False

    with pytest.raises(NotImplementedError):
        run(run_tool_loop(Bare("A1"), _ctx(), [_search_tool({"n": 0})]))


# ── Moderator / Session 集成 ────────────────────────────────────────────
def test_moderator_run_round_with_tools():
    counter = {"n": 0}
    reg = {"A1": MockAdapter("A1", tool_call="search", tool_args={"q": "x"})}
    mod = Moderator(reg)
    msgs = run(mod.run_round({"A1"}, _ctx(), tools=[_search_tool(counter)]))
    assert counter["n"] == 1 and "命中:x" in msgs[0].content


def test_session_single_mode_with_tools():
    counter = {"n": 0}
    reg = {"A1": MockAdapter("A1", tool_call="search", tool_args={"q": "架构"})}
    rt = Roundtable("tools-1", reg)
    answers = run(rt.ask_single("帮我查", tools=[_search_tool(counter)]))
    assert counter["n"] == 1
    assert rt.session.state == State.WAITING
    assert "命中:架构" in answers[0].content


def test_session_debate_with_tools():
    counter = {"n": 0}
    reg = {
        "A1": MockAdapter("A1", stance_seed="单体", tool_call="search", tool_args={"q": "成本"}),
        "A2": MockAdapter("A2", stance_seed="微服务", concede_at=1, concede_to="A1"),
    }
    rt = Roundtable("tools-debate", reg)
    run(rt.run_debate("辩", n_max=3, tools=[_search_tool(counter)]))
    assert counter["n"] >= 1                       # A1 每轮都会查
    assert rt.session.state == State.PROPOSAL


# ── 三家 provider 工具协议(合成响应,纯逻辑)──────────────────────────────
def test_claude_tool_hooks():
    a = ClaudeAdapter("A1")
    payload = a.with_tools(a.to_native(_ctx()), [_search_tool({"n": 0})])
    assert payload["tools"][0]["name"] == "search"
    assert "input_schema" in payload["tools"][0]
    native = {
        "content": [
            {"type": "text", "text": "让我查一下"},
            {"type": "tool_use", "id": "tu_1", "name": "search", "input": {"q": "架构"}},
        ],
        "stop_reason": "tool_use",
    }
    calls = a.parse_tool_calls(native)
    assert calls == [ToolCall("tu_1", "search", {"q": "架构"})]
    assert a.native_text(native) == "让我查一下"
    nxt = a.append_tool_round(payload, native, [ToolResult(calls[0], "命中")])
    assert nxt["messages"][-2]["role"] == "assistant"
    assert nxt["messages"][-1]["content"][0]["tool_use_id"] == "tu_1"


def test_gemini_tool_hooks():
    a = GeminiAdapter("A1")
    payload = a.with_tools(a.to_native(_ctx()), [_search_tool({"n": 0})])
    assert payload["tools"][0]["function_declarations"][0]["name"] == "search"
    native = {
        "candidates": [
            {"content": {"parts": [{"functionCall": {"name": "search", "args": {"q": "架构"}}}]}}
        ]
    }
    calls = a.parse_tool_calls(native)
    assert calls[0].name == "search" and calls[0].arguments == {"q": "架构"}
    nxt = a.append_tool_round(payload, native, [ToolResult(calls[0], "命中")])
    assert nxt["contents"][-1]["parts"][0]["functionResponse"]["response"]["result"] == "命中"


def test_grok_tool_hooks():
    a = GrokAdapter("A1")
    payload = a.with_tools(a.to_native(_ctx()), [_search_tool({"n": 0})])
    assert payload["tools"][0]["type"] == "function"
    assert payload["stream"] is False
    native = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {"name": "search", "arguments": '{"q": "架构"}'},
                        }
                    ],
                }
            }
        ]
    }
    calls = a.parse_tool_calls(native)
    assert calls == [ToolCall("call_1", "search", {"q": "架构"})]  # 字符串 arguments 已解析
    nxt = a.append_tool_round(payload, native, [ToolResult(calls[0], "命中")])
    assert nxt["messages"][-1] == {"role": "tool", "tool_call_id": "call_1", "content": "命中"}


if __name__ == "__main__":
    import subprocess
    raise SystemExit(subprocess.call(["pytest", "-q", __file__]))
