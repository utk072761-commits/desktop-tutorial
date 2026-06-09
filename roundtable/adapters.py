"""Adapter 层（蓝图 §2.2）——三家 provider 差异的封装地基。

真正的工作量在 adapter 层：三家 API 的 message 格式、tool use、
streaming 协议全不一样。这一层把差异封死，对外只暴露统一接口。

骨架里提供 MockAdapter（确定性、无网络），让整套流程立即可跑；
ClaudeAdapter / GeminiAdapter / GrokAdapter 的真实实现按相同契约填充
（见 §10.1：每个 adapter 必须带版本标记 + 契约测试）。
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import AsyncIterator, Dict, List

from .messages import InternalMessage


class AgentAdapter(ABC):
    """统一 adapter 契约。所有内部代码只依赖这三个方法。"""

    provider: str = "abstract"
    model: str = "abstract"
    api_version: str = "0"  # §10.1：版本标记，provider 协议变更时显式 bump
    supports_tools: bool = False  # tool_use 钩子是否实现（见 tools.run_tool_loop）

    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id

    @abstractmethod
    def to_native(self, msgs: List[InternalMessage]) -> dict:
        """内部格式 -> 该 provider 的请求 payload。"""

    @abstractmethod
    def from_native(self, resp: dict) -> InternalMessage:
        """provider 响应 -> 内部格式。"""

    @abstractmethod
    async def stream(self, payload: dict) -> AsyncIterator[str]:
        """统一的 token 流接口。"""

    async def respond(self, msgs: List[InternalMessage]) -> InternalMessage:
        """便捷封装：to_native -> stream 收集 -> from_native。

        Moderator 调用这一个方法即可，无需关心各 provider 的分帧差异。
        """
        payload = self.to_native(msgs)
        chunks: List[str] = []
        async for tok in self.stream(payload):
            chunks.append(tok)
        return self.from_native({"payload": payload, "text": "".join(chunks)})

    # ── tool_use 钩子（默认未实现；支持的 adapter 覆写并置 supports_tools=True）──
    def with_tools(self, payload: dict, tools: list) -> dict:
        """把工具声明注入请求 payload（provider 专属格式）。"""
        raise NotImplementedError

    async def call(self, payload: dict) -> dict:
        """非流式调用一次，返回 provider 原生响应 dict（tool loop 用）。"""
        raise NotImplementedError

    def parse_tool_calls(self, native: dict) -> list:
        """从 provider 响应抽工具调用，返回 list[ToolCall]。"""
        raise NotImplementedError

    def native_text(self, native: dict) -> str:
        """从 provider 响应抽最终文本。"""
        raise NotImplementedError

    def append_tool_round(self, payload: dict, native: dict, results: list) -> dict:
        """把模型的工具调用 + 工具结果追加进 payload，构造下一次请求。"""
        raise NotImplementedError


class MockAdapter(AgentAdapter):
    """确定性 mock：无需 API key，用于骨架端到端跑通与单测。

    每个 agent 有固定立场倾向（stance_seed），会在前几轮坚持、
    并按 `concede_at` 轮次接受他方论据以模拟收敛。输出严格遵循 §6
    的结构化字段格式，从而能被 `InternalMessage.parse_structured` 解析。
    """

    provider = "mock"
    api_version = "mock-1"
    supports_tools = True

    def __init__(
        self,
        agent_id: str,
        stance_seed: str = "保持中立",
        arguments: List[str] | None = None,
        concede_at: int | None = None,
        concede_to: str | None = None,
        latency: float = 0.0,
        fail: bool = False,
        tool_call: str | None = None,
        tool_args: dict | None = None,
    ) -> None:
        super().__init__(agent_id)
        self.model = f"mock-{agent_id}"
        self.stance_seed = stance_seed
        self.arguments = arguments or [f"{agent_id} 的核心论据 1", f"{agent_id} 的核心论据 2"]
        self.concede_at = concede_at
        self.concede_to = concede_to
        self.latency = latency
        # tool loop 模拟:若设了 tool_call,则在 loop 里先调一次该工具再给终稿。
        self.tool_call = tool_call
        self.tool_args = tool_args or {}
        self.fail = fail

    def to_native(self, msgs: List[InternalMessage]) -> dict:
        # mock 不真正请求；记录当前轮次与他方 agent，供生成回应用。
        cur_round = max((m.round for m in msgs), default=0)
        others = sorted(
            {m.agent_id for m in msgs if m.role == "agent" and m.agent_id != self.agent_id}
        )
        return {"round": cur_round + 1, "others": others}

    async def stream(self, payload: dict) -> AsyncIterator[str]:
        if self.fail:
            raise RuntimeError(f"{self.agent_id} provider 模拟故障")
        if self.latency:
            await asyncio.sleep(self.latency)
        for tok in self._compose(payload).split(" "):
            yield tok + " "

    def from_native(self, resp: dict) -> InternalMessage:
        text = resp["text"].strip()
        rnd = resp["payload"]["round"]
        msg = InternalMessage(role="agent", agent_id=self.agent_id, content=text, round=rnd)
        # 用结构化解析回填 refs，保证渲染连线与 REBUTS 一致。
        msg.refs = msg.parse_structured().rebuts
        return msg

    def _compose(self, payload: dict) -> str:
        rnd = payload["round"]
        others = payload["others"]
        conceding = (
            self.concede_at is not None
            and rnd >= self.concede_at
            and self.concede_to
        )
        if conceding:
            stance = f"我接受 @{self.concede_to} 的论据，调整立场趋同"
            args = [f"采纳 @{self.concede_to} 关于核心议题的论据", "据此更新我的方案"]
            rebuts = []
        else:
            stance = self.stance_seed
            args = self.arguments
            rebuts = [others[0]] if others else []

        rebut_line = ", ".join(rebuts) if rebuts else ""
        rebut_prose = (
            f"@{rebuts[0]} 你的第 1 条论据在该场景下不成立。" if rebuts else ""
        )
        note = payload.get("_results")
        note_prose = f"（已查询工具：{note}）" if note else ""
        body = f"{stance}。{note_prose}{rebut_prose}".strip()
        return (
            f"{body}\n"
            f"STANCE: {stance}\n"
            f"ARGS: {'; '.join(args)}\n"
            f"REBUTS: {rebut_line}"
        )

    # ── tool_use 钩子（确定性模拟，离线可跑）──────────────────────────
    def with_tools(self, payload: dict, tools: list) -> dict:
        payload = dict(payload)
        payload["_tools"] = [t.name for t in tools]
        return payload

    async def call(self, payload: dict) -> dict:
        if self.fail:
            raise RuntimeError(f"{self.agent_id} provider 模拟故障")
        if self.latency:
            await asyncio.sleep(self.latency)
        wants_tool = (
            self.tool_call
            and self.tool_call in payload.get("_tools", [])
            and not payload.get("_tool_done")
        )
        if wants_tool:
            return {
                "type": "tool_call",
                "calls": [
                    {"id": f"call-{self.agent_id}", "name": self.tool_call, "args": dict(self.tool_args)}
                ],
            }
        return {"type": "text", "text": self._compose(payload)}

    def parse_tool_calls(self, native: dict) -> list:
        from .tools import ToolCall

        if native.get("type") != "tool_call":
            return []
        return [ToolCall(c["id"], c["name"], c.get("args", {})) for c in native["calls"]]

    def native_text(self, native: dict) -> str:
        return native.get("text", "")

    def append_tool_round(self, payload: dict, native: dict, results: list) -> dict:
        payload = dict(payload)
        payload["_tool_done"] = True
        payload["_results"] = " | ".join(r.content for r in results)
        return payload
