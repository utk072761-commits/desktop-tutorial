"""tool_use:让参会模型在发言前调用工具（蓝图 §2.2 提到的 tool_use 接入）。

三家 provider 的工具协议各不相同（Claude 的 tool_use/tool_result 块、
Gemini 的 functionCall/functionResponse、OpenAI 兼容的 tool_calls/role:tool），
差异同样封死在各自 adapter 内（与 §2.1 一致）。本模块提供:

  - Tool / ToolCall / ToolResult —— provider 无关的统一工具类型;
  - run_tool_loop —— provider 无关的 agentic 循环骨架,只调用 adapter 暴露的
    五个小钩子(with_tools / call / parse_tool_calls / native_text / append_tool_round),
    每个钩子都是可单测的纯函数(call 除外,它是惟一的网络出口)。

循环本身可离线跑通:MockAdapter 实现同一套钩子,模拟「先调一次工具,再给终稿」。
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Union

from .messages import InternalMessage

# 工具处理函数:接收参数 dict,返回字符串结果(可同步或异步)。
ToolHandler = Callable[[Dict[str, Any]], Union[str, Awaitable[str]]]


@dataclass
class Tool:
    """provider 无关的工具声明 + 执行器。"""

    name: str
    description: str
    input_schema: Dict[str, Any]      # JSON Schema
    handler: ToolHandler

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Tool.name 不可为空")


@dataclass
class ToolCall:
    """模型发起的一次工具调用。"""

    id: str
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    """一次工具调用的执行结果。"""

    call: ToolCall
    content: str
    is_error: bool = False


async def execute(tool: Tool, call: ToolCall) -> ToolResult:
    """执行单个工具,异常隔离为 is_error 结果(不炸整轮)。"""
    try:
        out = tool.handler(call.arguments)
        if inspect.isawaitable(out):
            out = await out
        return ToolResult(call=call, content=str(out))
    except Exception as e:  # 工具失败降级为错误结果,交回模型决定如何处理
        return ToolResult(call=call, content=f"工具执行失败: {e}", is_error=True)


async def run_tool_loop(
    adapter,
    ctx: List[InternalMessage],
    tools: List[Tool],
    max_iters: int = 6,
) -> InternalMessage:
    """provider 无关的 agentic 循环:模型→(工具调用→结果)*→终稿。

    max_iters 是工具往返的硬上限(防模型反复调工具不收口);到顶则用最后一次
    响应的文本作为终稿。返回值是统一的 InternalMessage(refs 由结构化字段回填)。
    """
    if not getattr(adapter, "supports_tools", False):
        raise NotImplementedError(f"{type(adapter).__name__} 未实现 tool_use")

    by_name = {t.name: t for t in tools}
    payload = adapter.with_tools(adapter.to_native(ctx), tools)

    native: Any = None
    trace: List[Dict[str, Any]] = []  # 工具调用轨迹,挂到终稿 InternalMessage 上供 UI/审计
    for _ in range(max_iters):
        native = await adapter.call(payload)
        calls = adapter.parse_tool_calls(native)
        if not calls:
            break  # 无工具调用 => 这就是终稿

        results: List[ToolResult] = []
        for c in calls:
            tool = by_name.get(c.name)
            if tool is None:
                res = ToolResult(c, f"未知工具: {c.name}", is_error=True)
            else:
                res = await execute(tool, c)
            results.append(res)
            trace.append({
                "name": c.name,
                "arguments": c.arguments,
                "result": res.content,
                "is_error": res.is_error,
            })
        payload = adapter.append_tool_round(payload, native, results)

    text = adapter.native_text(native) if native is not None else ""
    msg = adapter.from_native({"payload": payload, "text": text})
    msg.tool_trace = trace
    return msg
