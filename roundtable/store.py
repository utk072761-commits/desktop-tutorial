"""持久化后端（蓝图 §8）。

蓝图:「单机起步用 SQLite/JSON 即可;并发与重试需求上来再换 Redis + 任务队列。」

本模块提供:
  - session_to_dict / session_from_dict —— Session ⇄ 纯 dict(可直接 json.dumps),
    UI 的 /api/state 与 JSON 落盘都复用它;
  - SessionStore —— 基于 stdlib sqlite3 的零依赖持久化(save/load/list/delete)。

`messages` 全量留存做审计/回放;`summaries` 喂下一轮。两者都落盘。
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import asdict
from typing import List, Optional

from .messages import (
    ConvergenceVerdict,
    Dispute,
    DisputeMatrix,
    InternalMessage,
    RoundSummary,
)
from .session import Session
from .state import State


# ── 序列化:Session -> dict(纯 JSON 友好)─────────────────────────────────
def session_to_dict(s: Session) -> dict:
    return {
        "session_id": s.session_id,
        "state": s.state.value,
        "active_set": sorted(s.active_set),
        "debate_mode": s.debate_mode,
        "round_counter": s.round_counter,
        "messages": [asdict(m) for m in s.messages],
        "summaries": [asdict(rs) for rs in s.summaries],
        "verdict": asdict(s.verdict) if s.verdict else None,
        "dispute_matrix": _matrix_to_dict(s.dispute_matrix) if s.dispute_matrix else None,
        "token_ledger": s.token_ledger,
    }


def _matrix_to_dict(m: DisputeMatrix) -> dict:
    return {
        "agreement": list(m.agreement),
        "disputes": [asdict(d) for d in m.disputes],
        "unexplored": list(m.unexplored),
    }


# ── 反序列化:dict -> Session ────────────────────────────────────────────
def session_from_dict(d: dict) -> Session:
    s = Session(session_id=d["session_id"])
    s.state = State(d["state"])
    s.active_set = set(d.get("active_set", []))
    s.debate_mode = d.get("debate_mode", False)
    s.round_counter = d.get("round_counter", 0)
    s.messages = [InternalMessage(**m) for m in d.get("messages", [])]
    s.summaries = [RoundSummary(**rs) for rs in d.get("summaries", [])]
    v = d.get("verdict")
    s.verdict = ConvergenceVerdict(**v) if v else None
    dm = d.get("dispute_matrix")
    s.dispute_matrix = _matrix_from_dict(dm) if dm else None
    s.token_ledger = d.get("token_ledger", {})
    return s


def _matrix_from_dict(d: dict) -> DisputeMatrix:
    return DisputeMatrix(
        agreement=list(d.get("agreement", [])),
        disputes=[Dispute(**x) for x in d.get("disputes", [])],
        unexplored=list(d.get("unexplored", [])),
    )


# ── SQLite 持久化 ───────────────────────────────────────────────────────
class SessionStore:
    """零依赖会话存储。整条 Session 序列化为 JSON 存一行,便于审计/回放。"""

    def __init__(self, path: str = "roundtable.db") -> None:
        self.path = path
        self._conn = sqlite3.connect(path)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                state      TEXT NOT NULL,
                data       TEXT NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )
        self._conn.commit()

    def save(self, session: Session) -> None:
        self._conn.execute(
            "REPLACE INTO sessions (session_id, state, data, updated_at) VALUES (?, ?, ?, ?)",
            (
                session.session_id,
                session.state.value,
                json.dumps(session_to_dict(session), ensure_ascii=False),
                time.time(),
            ),
        )
        self._conn.commit()

    def load(self, session_id: str) -> Optional[Session]:
        row = self._conn.execute(
            "SELECT data FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        return session_from_dict(json.loads(row[0])) if row else None

    def list_sessions(self) -> List[str]:
        rows = self._conn.execute(
            "SELECT session_id FROM sessions ORDER BY updated_at DESC"
        ).fetchall()
        return [r[0] for r in rows]

    def delete(self, session_id: str) -> None:
        self._conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
