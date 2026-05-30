"""端到端编排：首板初筛 → 强中择优 → 次日进场观察。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd

from config import Config, DEFAULT_CONFIG
from . import datasource as ds
from .screening import screen_first_boards, ScreenResult
from .ranking import rank_picks, RankedPick
from .entry import check_auction, AuctionCheck


@dataclass
class PipelineOutput:
    date: str
    pool_size: int
    first_board_size: int
    screened: List[ScreenResult] = field(default_factory=list)   # 全部首板的初筛结论
    passed: List[ScreenResult] = field(default_factory=list)     # 通过七步初筛
    watchlist: List[RankedPick] = field(default_factory=list)    # PK 后的 2~3 只自选

    def summary(self) -> str:
        lines = [
            f"=== 首板复盘 {self.date} ===",
            f"涨停池 {self.pool_size} 只，其中首板 {self.first_board_size} 只；"
            f"通过七步初筛 {len(self.passed)} 只，锁定自选 {len(self.watchlist)} 只。",
            "",
            "── 自选池（强中择优）──",
        ]
        if not self.watchlist:
            lines.append("（今日无符合条件的标的）")
        for i, p in enumerate(self.watchlist, 1):
            lines.append(
                f"{i}. {p.code} {p.name}  总分={p.score} "
                f"[封板时间{p.seal_time_score} 量价{p.volume_price_score} 力度{p.seal_strength_score}]"
            )
            lines.append(f"     {p.detail}")
        # 被否决标的的原因（便于复盘）
        rejected = [r for r in self.screened if not r.passed]
        if rejected:
            lines += ["", f"── 被否决 {len(rejected)} 只（节选原因）──"]
            for r in rejected[:20]:
                lines.append(f"  {r.code} {r.name}: {'; '.join(r.reasons)}")
        return "\n".join(lines)


class FirstBoardPipeline:
    def __init__(self, config: Config = DEFAULT_CONFIG):
        self.config = config

    def run(self, date: str, use_history: bool = True,
            confirm_burst: bool = True) -> PipelineOutput:
        """对首板当日（date='YYYYMMDD'）收盘后复盘。"""
        pool = ds.get_zt_pool(date)
        first = pool[pool["连板数"] == 1] if not pool.empty else pool

        screened = screen_first_boards(date, pool=pool, config=self.config,
                                       use_history=use_history)
        passed = [r for r in screened if r.passed]
        watchlist = rank_picks(date, passed, config=self.config,
                               confirm_burst=confirm_burst)
        return PipelineOutput(
            date=date,
            pool_size=len(pool),
            first_board_size=len(first),
            screened=screened,
            passed=passed,
            watchlist=watchlist,
        )

    def next_day_auction(self, watchlist: List[RankedPick],
                         board_date: str) -> List[AuctionCheck]:
        """次日开盘前，对自选池逐只做集合竞价检查。

        需要昨日（board_date）收盘价/换手率/流通市值——从涨停池回取。
        """
        pool = ds.get_zt_pool(board_date)
        pool = pool.set_index("代码") if not pool.empty else pool
        checks: List[AuctionCheck] = []
        for p in watchlist:
            prev_close = prev_to = float_shares = None
            if not pool.empty and p.code in pool.index:
                row = pool.loc[p.code]
                prev_close = float(row.get("最新价") or 0)
                prev_to = float(row.get("换手率") or 0)
                fmv = float(row.get("流通市值") or 0)
                float_shares = fmv / prev_close if prev_close else None
            checks.append(check_auction(
                p.code, p.name, prev_close, prev_to, float_shares, self.config))
        return checks


def run_pipeline(date: str, config: Config = DEFAULT_CONFIG,
                 use_history: bool = True) -> PipelineOutput:
    """便捷入口。"""
    return FirstBoardPipeline(config).run(date, use_history=use_history)
