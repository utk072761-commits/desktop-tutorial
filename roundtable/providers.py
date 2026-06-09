"""真实 provider adapter(蓝图 §2.2 / §10.1)。

把三家 API 的差异封死在各自 adapter 内,对外只暴露 AgentAdapter 契约。
依赖(anthropic / httpx)全部**惰性导入**——不装也能 import 本模块、跑 MockAdapter
与全部离线单测;只有真正发起网络调用时才需要。

可验证、可契约测试的部分(§10.1)都是纯函数:
  - to_native     内部消息 -> provider 请求 payload(角色映射在此)
  - from_native   provider 文本 -> 内部消息
  - _extract_*    各家 streaming 分帧里抽 token 的逻辑

真正的 HTTP/SDK 调用集中在 stream() 里,薄且可替换。每个 adapter 带 api_version
版本标记(§10.1:provider 协议变更时显式 bump,并跑契约测试)。

模型默认:Claude 用 claude-opus-4-8;裁判/压缩等轻量场景见 llm.py 用 claude-haiku-4-5。
"""

from __future__ import annotations

import json
from typing import AsyncIterator, Dict, List

from .adapters import AgentAdapter
from .messages import InternalMessage
from .session import SYSTEM_PROMPT


def _role_for(agent_id: str, self_id: str) -> str:
    """本方发言算 assistant,其余(用户/他方/主持人)都算 user 输入。"""
    return "assistant" if agent_id == self_id else "user"


def _label(msg: InternalMessage, self_id: str) -> str:
    """非本方发言带上说话人标签,让模型清楚谁说了什么。"""
    if msg.agent_id == self_id:
        return msg.content
    return f"[{msg.agent_id}]: {msg.content}"


# ════════════════════════════════════════════════════════════════════════
# Claude —— 官方 anthropic SDK(AsyncAnthropic + messages.stream)
# ════════════════════════════════════════════════════════════════════════
class ClaudeAdapter(AgentAdapter):
    provider = "anthropic"
    api_version = "2023-06-01"

    def __init__(
        self,
        agent_id: str,
        model: str = "claude-opus-4-8",
        system: str = SYSTEM_PROMPT,
        api_key: str | None = None,
        max_tokens: int = 1024,
    ) -> None:
        super().__init__(agent_id)
        self.model = model
        self.system = system
        self.api_key = api_key
        self.max_tokens = max_tokens

    def to_native(self, msgs: List[InternalMessage]) -> dict:
        cur_round = max((m.round for m in msgs), default=0)
        # Claude 只有 user/assistant 两个对话角色,system 单独传。
        messages = [
            {"role": _role_for(m.agent_id, self.agent_id), "content": _label(m, self.agent_id)}
            for m in msgs
            if m.role != "system"
        ]
        if not messages or messages[0]["role"] != "user":
            # 首条必须是 user;辩论上下文恒以用户原始问题开头,这里兜底。
            messages.insert(0, {"role": "user", "content": "(开始)"})
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": self.system,
            "messages": messages,
            "_round": cur_round + 1,
        }

    def from_native(self, resp: dict) -> InternalMessage:
        rnd = resp.get("payload", {}).get("_round", 1)
        msg = InternalMessage(
            role="agent", agent_id=self.agent_id, content=resp["text"].strip(), round=rnd
        )
        msg.refs = msg.parse_structured().rebuts
        return msg

    async def stream(self, payload: dict) -> AsyncIterator[str]:
        import anthropic  # 惰性导入

        client = anthropic.AsyncAnthropic(api_key=self.api_key)
        kwargs = {k: v for k, v in payload.items() if not k.startswith("_")}
        # Opus 4.8 已移除 temperature/top_p/budget_tokens——一律不传,否则 400。
        async with client.messages.stream(**kwargs) as s:
            async for token in s.text_stream:
                yield token


# ════════════════════════════════════════════════════════════════════════
# Gemini —— REST streamGenerateContent(SSE),避免猜 SDK 绑定
# ════════════════════════════════════════════════════════════════════════
class GeminiAdapter(AgentAdapter):
    provider = "google"
    api_version = "v1beta"

    def __init__(
        self,
        agent_id: str,
        model: str = "gemini-2.5-pro",
        system: str = SYSTEM_PROMPT,
        api_key: str | None = None,
    ) -> None:
        super().__init__(agent_id)
        self.model = model
        self.system = system
        self.api_key = api_key

    def to_native(self, msgs: List[InternalMessage]) -> dict:
        cur_round = max((m.round for m in msgs), default=0)
        # Gemini 角色:本方 "model",他方 "user";system 走 system_instruction。
        contents = [
            {
                "role": "model" if m.agent_id == self.agent_id else "user",
                "parts": [{"text": _label(m, self.agent_id)}],
            }
            for m in msgs
            if m.role != "system"
        ]
        if not contents or contents[0]["role"] != "user":
            contents.insert(0, {"role": "user", "parts": [{"text": "(开始)"}]})
        return {
            "system_instruction": {"parts": [{"text": self.system}]},
            "contents": contents,
            "_round": cur_round + 1,
        }

    def from_native(self, resp: dict) -> InternalMessage:
        rnd = resp.get("payload", {}).get("_round", 1)
        msg = InternalMessage(
            role="agent", agent_id=self.agent_id, content=resp["text"].strip(), round=rnd
        )
        msg.refs = msg.parse_structured().rebuts
        return msg

    @staticmethod
    def _extract_text(chunk: dict) -> str:
        """从 streamGenerateContent 单个 SSE chunk 抽文本。"""
        out = []
        for cand in chunk.get("candidates", []):
            for part in cand.get("content", {}).get("parts", []):
                if "text" in part:
                    out.append(part["text"])
        return "".join(out)

    async def stream(self, payload: dict) -> AsyncIterator[str]:
        import httpx  # 惰性导入

        body = {k: v for k, v in payload.items() if not k.startswith("_")}
        url = (
            f"https://generativelanguage.googleapis.com/{self.api_version}"
            f"/models/{self.model}:streamGenerateContent"
        )
        params = {"alt": "sse", "key": self.api_key}
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", url, params=params, json=body) as r:
                async for data in _iter_sse(r.aiter_lines()):
                    if data == "[DONE]":
                        break
                    text = self._extract_text(json.loads(data))
                    if text:
                        yield text


# ════════════════════════════════════════════════════════════════════════
# Grok(xAI)—— OpenAI 兼容 /v1/chat/completions(SSE)
# ════════════════════════════════════════════════════════════════════════
class GrokAdapter(AgentAdapter):
    provider = "xai"
    api_version = "v1"

    def __init__(
        self,
        agent_id: str,
        model: str = "grok-4",
        system: str = SYSTEM_PROMPT,
        api_key: str | None = None,
        base_url: str = "https://api.x.ai/v1",
    ) -> None:
        super().__init__(agent_id)
        self.model = model
        self.system = system
        self.api_key = api_key
        self.base_url = base_url

    def to_native(self, msgs: List[InternalMessage]) -> dict:
        cur_round = max((m.round for m in msgs), default=0)
        # OpenAI 兼容:system 一条 + 本方 assistant / 他方 user。
        messages = [{"role": "system", "content": self.system}]
        messages += [
            {"role": _role_for(m.agent_id, self.agent_id), "content": _label(m, self.agent_id)}
            for m in msgs
            if m.role != "system"
        ]
        return {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "_round": cur_round + 1,
        }

    def from_native(self, resp: dict) -> InternalMessage:
        rnd = resp.get("payload", {}).get("_round", 1)
        msg = InternalMessage(
            role="agent", agent_id=self.agent_id, content=resp["text"].strip(), round=rnd
        )
        msg.refs = msg.parse_structured().rebuts
        return msg

    @staticmethod
    def _extract_text(chunk: dict) -> str:
        """从 chat.completions 流式 chunk 抽 delta 文本。"""
        out = []
        for choice in chunk.get("choices", []):
            piece = choice.get("delta", {}).get("content")
            if piece:
                out.append(piece)
        return "".join(out)

    async def stream(self, payload: dict) -> AsyncIterator[str]:
        import httpx  # 惰性导入

        body = {k: v for k, v in payload.items() if not k.startswith("_")}
        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST", f"{self.base_url}/chat/completions", headers=headers, json=body
            ) as r:
                async for data in _iter_sse(r.aiter_lines()):
                    if data == "[DONE]":
                        break
                    text = self._extract_text(json.loads(data))
                    if text:
                        yield text


# ── 统一 SSE 分帧 ───────────────────────────────────────────────────────
async def _iter_sse(lines: AsyncIterator[str]) -> AsyncIterator[str]:
    """从 `data: ...` 行流里抽 payload(忽略注释、空行、event 行)。"""
    async for line in lines:
        line = line.strip()
        if not line or line.startswith(":") or not line.startswith("data:"):
            continue
        yield line[len("data:"):].strip()


PROVIDERS = {
    "anthropic": ClaudeAdapter,
    "google": GeminiAdapter,
    "xai": GrokAdapter,
}
