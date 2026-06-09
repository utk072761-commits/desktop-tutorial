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
pytest tests/test_roundtable.py -q  # 17 项单测，覆盖各核心不变量
```

## 模块 ↔ 蓝图对照

| 文件 | 蓝图章节 | 职责 |
|------|---------|------|
| `messages.py`   | §2.1 / §3.2 / §4 / §5 | 内部统一消息格式 + 各结构化数据类 |
| `state.py`      | §1   | 闭合状态机（显式转换表，非法转换抛错，全局中断边） |
| `adapters.py`   | §2.2 | Adapter 契约 + MockAdapter（真实 provider 在此扩展） |
| `dispatcher.py` | §2.3 | `@` 点名解析与激活子集 |
| `moderator.py`  | §3   | 并发调用（取代 mutex）+ context 压缩 + turn/token 双闸门 |
| `judge.py`      | §4   | 外置一致性裁判（不参与辩论，只做收敛判定） |
| `decision.py`   | §5   | 分歧矩阵（各方最强论点对立，非共识摘要） |
| `session.py`    | §1 / §6 / §8 | 会话状态存储 + 顶层编排器 + 协作协议 system prompt |

## 守住的物理红线

- **副作用只挂在 `COMMITTED` 之后**（§7.3）：`Roundtable.approve()` 是唯一
  允许执行有副作用动作的入口，且只有状态合法跨过 `PROPOSAL → COMMITTED`
  后才会触发。这是"中书省 vs 邮局"的物理分界线。
- **收敛由外置裁判判定**（§4），不由参与者自己宣布。
- **喂模型用 `summaries`，不用 `messages`**（§8）：全量发言只做审计/回放，
  下一轮 context 只传上一轮摘要 + 用户原始问题，避免 context 线性膨胀。

## 尚未实现（骨架边界）

- 真实三家 provider adapter（auth / tool_use / streaming 分帧）——见 §10.1 契约测试要求；
- UI 三区（圆桌视窗 / 决策板 / 行动控制台），§7；
- 持久化后端（当前为内存 `Session` 对象，§8 建议 SQLite/JSON 起步）；
- 裁判/压缩改用真实轻量模型（当前为确定性启发式）。
