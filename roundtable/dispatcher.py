"""点名与激活子集（蓝图 §2.3）。

@ 解析本身是小事，蓝图反复强调真正的工作量在 adapter 层；
这里只做最朴素、可测的实现。
"""

from __future__ import annotations

import re
from typing import Dict, Set

from .adapters import AgentAdapter

# 形如 @A1 / @A12 / @judge 的点名标记。
_MENTION_RE = re.compile(r"@([A-Za-z][A-Za-z0-9_]*)")


def parse_mentions(prompt: str) -> Set[str]:
    """从用户 prompt 提取被 @ 的 agent_id 集合。"""
    return set(_MENTION_RE.findall(prompt))


def resolve_active_set(prompt: str, registry: Dict[str, AgentAdapter]) -> Set[str]:
    """解析激活子集 S：有 @ 取交集，无 @ 则全员激活（§2.3）。

    被 @ 但不在 registry 的标签忽略（例如误点了未注册的成员）。
    """
    mentioned = parse_mentions(prompt) & set(registry)
    return mentioned if mentioned else set(registry)
