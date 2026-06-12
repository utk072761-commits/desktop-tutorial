"""真实轻量模型驱动的裁判与压缩(蓝图 §3.2 / §4 / §10.2)。

骨架默认用确定性启发式(judge.Judge / InternalMessage.parse_structured);
本模块给出「换成真实轻量模型」的实现,保持同一接口可直接替换:

  - LLMJudge      外置裁判(§4):读各方 RoundSummary,产出结构化 ConvergenceVerdict。
  - LLMCompressor context 压缩(§3.2):把一条发言压成 RoundSummary。

两者都接受一个通用异步补全函数 `complete(system, user) -> str`(应返回 JSON 文本),
因此可被任意 provider 适配。`make_claude_completer` 给出基于官方 anthropic SDK +
结构化输出的真实实现,默认用 claude-haiku-4-5(轻量、便宜,适合裁判/压缩)。

裁判要「低 temperature + 结构化输出约束」(§10.2)。Opus/Haiku 4.x 已移除 temperature,
故改用**结构化输出**(output_config.format)约束收敛性,等价地获得稳定结构。
"""

from __future__ import annotations

import json
from typing import Awaitable, Callable, List

from .messages import ConvergenceVerdict, RoundSummary

# 通用补全函数签名:给定 system + user 提示,返回(JSON)文本。
Completer = Callable[[str, str], Awaitable[str]]


_JUDGE_SYSTEM = (
    "你是多智能体圆桌会议的独立裁判。你不参与辩论,不发表观点,"
    "只判定各方是否已经收敛。严格依据给定的各方立场摘要,输出 JSON。"
)

_JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "converged": {"type": "boolean"},
        "agreement_points": {"type": "array", "items": {"type": "string"}},
        "open_disputes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "positions": {"type": "object", "additionalProperties": {"type": "string"}},
                },
                "required": ["topic", "positions"],
                "additionalProperties": False,
            },
        },
        "confidence": {"type": "number"},
    },
    "required": ["converged", "agreement_points", "open_disputes", "confidence"],
    "additionalProperties": False,
}

_SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "stance": {"type": "string"},
        "key_arguments": {"type": "array", "items": {"type": "string"}},
        "rebuts": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["stance", "key_arguments", "rebuts"],
    "additionalProperties": False,
}


class LLMJudge:
    """外置裁判,把收敛判定交给独立模型(§4)。接口与 judge.Judge 一致(.verdict)。"""

    def __init__(self, complete: Completer) -> None:
        self._complete = complete

    async def verdict(self, summaries: List[RoundSummary]) -> ConvergenceVerdict:
        if not summaries:
            return ConvergenceVerdict(converged=False, confidence=0.0)
        user = "各方本轮立场摘要:\n" + "\n".join(s.render() for s in summaries)
        raw = await self._complete(_JUDGE_SYSTEM, user)
        data = json.loads(raw)
        return ConvergenceVerdict(
            converged=bool(data["converged"]),
            agreement_points=list(data.get("agreement_points", [])),
            open_disputes=list(data.get("open_disputes", [])),
            confidence=float(data.get("confidence", 0.0)),
        )


class LLMCompressor:
    """用轻量模型把一条发言压成 RoundSummary(§3.2),替代自带结构化字段路径。"""

    def __init__(self, complete: Completer) -> None:
        self._complete = complete

    async def summarize(self, agent_id: str, content: str, round: int = 0) -> RoundSummary:
        system = "把以下发言压成结构化立场摘要,输出 JSON(stance/key_arguments/rebuts)。"
        raw = await self._complete(system, content)
        data = json.loads(raw)
        return RoundSummary(
            agent_id=agent_id,
            stance=data.get("stance", ""),
            key_arguments=list(data.get("key_arguments", [])),
            rebuts=list(data.get("rebuts", [])),
            round=round,
        )


# ── 真实 Claude 补全器(结构化输出)────────────────────────────────────
def make_claude_completer(
    model: str = "claude-haiku-4-5",
    schema: dict | None = None,
    api_key: str | None = None,
    max_tokens: int = 1024,
) -> Completer:
    """构造一个用官方 anthropic SDK + 结构化输出约束的补全函数(惰性导入)。

    schema 默认用裁判 schema;压缩场景传 _SUMMARY_SCHEMA。
    Haiku 4.5 支持结构化输出但不支持 effort 参数,故只传 output_config.format。
    """
    fmt = {"type": "json_schema", "schema": schema or _JUDGE_SCHEMA}

    async def _complete(system: str, user: str) -> str:
        import anthropic  # 惰性导入

        client = anthropic.AsyncAnthropic(api_key=api_key)
        resp = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"format": fmt},
        )
        # 结构化输出保证首个 text block 是合法 JSON。
        return next(b.text for b in resp.content if b.type == "text")

    return _complete


def claude_judge(model: str = "claude-haiku-4-5", api_key: str | None = None) -> LLMJudge:
    return LLMJudge(make_claude_completer(model=model, schema=_JUDGE_SCHEMA, api_key=api_key))


def claude_compressor(model: str = "claude-haiku-4-5", api_key: str | None = None) -> LLMCompressor:
    return LLMCompressor(make_claude_completer(model=model, schema=_SUMMARY_SCHEMA, api_key=api_key))
