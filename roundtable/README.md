# 中书省 · 多智能体圆桌会议系统（骨架 v1.0）

按蓝图 v1.0 落地的**可运行骨架**。一个由用户终审的多模型协作系统：
模型负责拟稿、辩论、复核；用户保留封驳权与画敕权。

骨架默认使用 `MockAdapter`（确定性输出、零依赖、无需任何 API key），
因此 `python run_roundtable.py` 与 `pytest tests/test_roundtable.py` 立即可跑。
接真实模型时，只需按 `AgentAdapter` 契约实现 `ClaudeAdapter` / `GeminiAdapter`
/ `GrokAdapter`，其余各层无需改动。

## 跑起来

```bash
python run_roundtable.py            # 端到端演示：单问 -> 辩论收敛 -> 分歧矩阵 -> 终审
python -m roundtable.ui.server      # 决策工作台 UI（浏览器打开 http://127.0.0.1:8000）
pytest tests/ -q   # 53 项单测（核心 + 边界扩展 + tool_use + 流式/中断）
```

核心包零依赖。真实 provider 调用是可选项：`pip install anthropic httpx`
（Claude 走官方 SDK，Gemini/Grok 走各自 REST/OpenAI 兼容端点）。不装也能跑通
MockAdapter 驱动的全部演示与单测。

## 模块 ↔ 蓝图对照

| 文件 | 蓝图章节 | 职责 |
|------|---------|------|
| `messages.py`   | §2.1 / §3.2 / §4 / §5 | 内部统一消息格式 + 各结构化数据类 |
| `state.py`      | §1   | 闭合状态机（显式转换表，非法转换抛错，全局中断边） |
| `adapters.py`   | §2.2 | Adapter 契约 + MockAdapter |
| `providers.py`  | §2.2 / §10.1 | 真实 ClaudeAdapter / GeminiAdapter / GrokAdapter（带版本标记 + 契约测试） |
| `dispatcher.py` | §2.3 | `@` 点名解析与激活子集 |
| `moderator.py`  | §3   | 并发调用（取代 mutex）+ context 压缩 + turn/token 双闸门 |
| `tools.py`      | §2.2 | tool_use:统一 Tool/ToolCall/ToolResult + provider 无关 agentic 循环 |
| `judge.py`      | §4   | 外置一致性裁判（确定性启发式） |
| `llm.py`        | §4 / §3.2 / §10.2 | 真实轻量模型裁判 LLMJudge + 压缩 LLMCompressor（Claude 结构化输出，默认 Haiku） |
| `decision.py`   | §5   | 分歧矩阵（各方最强论点对立，非共识摘要） |
| `session.py`    | §1 / §6 / §8 | 会话状态存储 + 顶层编排器 + 协作协议 system prompt |
| `store.py`      | §8   | 持久化（SQLite + Session ⇄ dict 序列化） |
| `ui/`           | §7   | 决策工作台（圆桌视窗 / 决策板 / 行动控制台 + tool_use 轨迹 + 逐 token 实时渲染 + 介入辩论，stdlib 零依赖） |

## 守住的物理红线

- **副作用只挂在 `COMMITTED` 之后**（§7.3）：`Roundtable.approve()` 是唯一
  允许执行有副作用动作的入口，且只有状态合法跨过 `PROPOSAL → COMMITTED`
  后才会触发。这是"中书省 vs 邮局"的物理分界线。
- **收敛由外置裁判判定**（§4），不由参与者自己宣布。
- **喂模型用 `summaries`，不用 `messages`**（§8）：全量发言只做审计/回放，
  下一轮 context 只传上一轮摘要 + 用户原始问题，避免 context 线性膨胀。

## 已补齐的骨架边界

- ✅ 真实三家 provider adapter（`providers.py`）：auth、角色映射、streaming 分帧，
  各带 `api_version` 版本标记，纯逻辑部分有契约测试（§10.1）。
- ✅ tool_use（`tools.py`）：统一 `Tool/ToolCall/ToolResult` + provider 无关的 agentic
  循环（模型→工具调用→结果→终稿，带往返硬上限）。三家工具协议
  （Claude tool_use 块 / Gemini functionCall / OpenAI 兼容 tool_calls）封在各 adapter
  的五个纯钩子里，逐一有契约测试；`Moderator.run_round` / `Session.ask_single` /
  `run_debate` 都可传 `tools=[...]`，单模型工具失败/未知工具均隔离不炸整轮。
  工具调用轨迹记在 `InternalMessage.tool_trace`（随会话序列化），UI 圆桌视窗
  以 `🔧 search({...}) → 结果` 形式逐条呈现（勾选「启用工具」即可）。
- ✅ UI 三区（`ui/`）：圆桌视窗 / 决策板 / 行动控制台，与后端状态机绑定（§7）。
- ✅ 持久化后端（`store.py`）：SQLite + JSON 友好序列化（§8）。
- ✅ 裁判/压缩真实轻量模型路径（`llm.py`）：Claude 结构化输出，默认 `claude-haiku-4-5`；
  `Roundtable` 同时兼容同步启发式裁判与异步 LLM 裁判。
- ✅ 流式实时渲染 + 介入辩论（§3.1/§7.3）：`on_token` 逐 token 回调贯穿
  adapter→moderator→session；UI 辩论转后台线程，前端轮询 `live` 缓冲实时呈现
  各方生成中的发言（每轮落定即清空，最终仍按固定座次落座）；`run_debate`
  新增 `interrupt` 探针——UI「介入辩论」按钮经此走真实的
  DISCUSSION --user_interrupt--> PROPOSAL 边（busy 期间「全局中断」也路由到此，
  避免与后台轮转换竞争）。

## 仍待真实环境验证

- provider adapter 的 streaming 与 tool_use 仅在合成数据上做了契约测试，未对活 API
  跑过——§10.1 要求接活后补端到端契约测试，provider 协议变更时 bump `api_version`。
- 并发与重试压力上来后，§8 建议把 SQLite 换成 Redis + 任务队列。
