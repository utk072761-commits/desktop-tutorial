"""中书省 · 多智能体圆桌会议系统（骨架实现 v1.0）。

按蓝图 §1–§8 落地的可运行骨架：

  - messages   §2.1 内部统一消息格式 + §3.2/§4/§5 各结构化数据类
  - state      §1 闭合状态机
  - adapters   §2.2 Adapter 接口 + Mock 实现（真实 provider 在此扩展）
  - dispatcher §2.3 点名与激活子集
  - moderator  §3 并发调用 + context 压缩 + 双闸门
  - judge      §4 外置一致性裁判
  - decision   §5 分歧矩阵
  - session    §8 会话状态存储 + 顶层编排器

Adapter 默认用 MockAdapter（确定性输出，无需任何 API key），
因此整套流程 `python run_roundtable.py` 即可端到端跑通。
"""

from .messages import (
    InternalMessage,
    RoundSummary,
    ConvergenceVerdict,
    Dispute,
    DisputeMatrix,
)
from .state import State, Transition, StateMachine, IllegalTransition
from .adapters import AgentAdapter, MockAdapter
from .dispatcher import parse_mentions, resolve_active_set
from .moderator import Moderator, TokenLedger, BudgetExceeded
from .judge import Judge
from .decision import build_dispute_matrix
from .session import Session, Roundtable, SYSTEM_PROMPT
from .providers import ClaudeAdapter, GeminiAdapter, GrokAdapter, PROVIDERS
from .llm import LLMJudge, LLMCompressor, claude_judge, claude_compressor
from .store import SessionStore, session_to_dict, session_from_dict

__all__ = [
    "InternalMessage",
    "RoundSummary",
    "ConvergenceVerdict",
    "Dispute",
    "DisputeMatrix",
    "State",
    "Transition",
    "StateMachine",
    "IllegalTransition",
    "AgentAdapter",
    "MockAdapter",
    "parse_mentions",
    "resolve_active_set",
    "Moderator",
    "TokenLedger",
    "BudgetExceeded",
    "Judge",
    "build_dispute_matrix",
    "Session",
    "Roundtable",
    "SYSTEM_PROMPT",
    "ClaudeAdapter",
    "GeminiAdapter",
    "GrokAdapter",
    "PROVIDERS",
    "LLMJudge",
    "LLMCompressor",
    "claude_judge",
    "claude_compressor",
    "SessionStore",
    "session_to_dict",
    "session_from_dict",
]
