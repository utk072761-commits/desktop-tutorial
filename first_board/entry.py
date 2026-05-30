"""实战进场节点：次日博弈。

针对自选池中的 2~3 只标的：
  A) 集合竞价观察（09:15–09:25）：
       - 理想高开 3%~5%；
       - 竞价换手率 ≈ 昨日全天换手率的 10%（有资金抢筹）。
  B) 唯一进场信号——打板：
       绝不盘中追涨；只在资金扫货、即将封死涨停板的瞬间，挂涨停价打板。
       “先涨停、先符合条件，就打谁。”
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from config import Config, DEFAULT_CONFIG, limit_pct
from . import datasource as ds


@dataclass
class AuctionCheck:
    code: str
    name: str
    open_pct: Optional[float]          # 竞价高开幅度（%）
    auction_turnover: Optional[float]  # 竞价换手率（%）
    prev_turnover: Optional[float]     # 昨日全天换手率（%）
    turnover_ratio: Optional[float]    # 竞价换手 / 昨日换手
    qualified: bool
    notes: List[str]


def round_limit_price(prev_close: float, code: str) -> float:
    """涨停价 = 前收 × (1+涨幅)，四舍五入到分。"""
    return round(prev_close * (1 + limit_pct(code)) + 1e-9, 2)


# ── A) 集合竞价观察 ──────────────────────────────────────────────────────
def check_auction(
    code: str,
    name: str,
    prev_close: float,
    prev_turnover: float,
    float_shares: float,
    config: Config = DEFAULT_CONFIG,
) -> AuctionCheck:
    """次日开盘前/开盘瞬间评估集合竞价是否符合进场前提。

    prev_close    : 昨日（首板）收盘价。
    prev_turnover : 昨日全天换手率（%），来自涨停池“换手率”。
    float_shares  : 流通股本（股）= 昨日流通市值 / 昨日收盘价。
    """
    ec = config.entry
    notes: List[str] = []
    df = ds.get_pre_auction(code)
    open_pct = auction_to = ratio = None

    if not df.empty:
        row925 = df[df["time"].astype(str).str.endswith("09:25:00")]
        last = row925.iloc[-1] if not row925.empty else df.iloc[-1]
        auction_px = float(last.get("close") or last.get("open") or 0)
        if prev_close:
            open_pct = (auction_px / prev_close - 1) * 100
        # 竞价成交量（手）→ 竞价换手率
        vol_hand = pd.to_numeric(last.get("volume", 0), errors="coerce")
        if float_shares:
            auction_to = float(vol_hand) * 100 / float_shares * 100  # 手→股
        if auction_to is not None and prev_turnover:
            ratio = auction_to / prev_turnover
    else:
        notes.append("盘前竞价数据不可用（非交易时段或接口受限）")

    qualified = True
    if open_pct is None:
        qualified = False
    else:
        if not (ec.auction_open_low <= open_pct <= ec.auction_open_high):
            qualified = False
            notes.append(f"高开{open_pct:.1f}% 不在理想区间[{ec.auction_open_low}%,{ec.auction_open_high}%]")
        if ratio is not None:
            lo = ec.auction_turnover_ratio * (1 - ec.auction_turnover_tol)
            hi = ec.auction_turnover_ratio * (1 + ec.auction_turnover_tol)
            if not (lo <= ratio <= hi):
                qualified = False
                notes.append(f"竞价换手比{ratio:.2f} 偏离目标~{ec.auction_turnover_ratio}")
        else:
            notes.append("竞价换手率无法计算（缺流通股本/昨日换手）")

    return AuctionCheck(code, name, open_pct, auction_to, prev_turnover, ratio,
                        qualified, notes)


# ── B) 打板进场信号 ──────────────────────────────────────────────────────
@dataclass
class EntrySignal:
    code: str
    fire: bool                 # 是否触发打板
    limit_price: float         # 应挂的涨停价
    reason: str
    snapshot: dict


def board_entry_signal(
    code: str,
    name: str,
    prev_close: float,
    config: Config = DEFAULT_CONFIG,
) -> EntrySignal:
    """盘中实时判定“即将封死涨停”的打板信号（仅交易时段有效）。

    触发条件：现价 ≥ 涨停价×seal_trigger_ratio 且买一处于涨停价位扫货
              （封单正在快速堆积）→ 立即挂涨停价。
    返回 fire=True 时，交易端应以 limit_price（涨停价）市价/限价打板。
    """
    lp = round_limit_price(prev_close, code)
    q = ds.get_realtime_quote(code)
    if not q:
        return EntrySignal(code, False, lp, "实时盘口不可用（非交易时段或接口受限）", {})

    def _f(key):
        try:
            return float(q.get(key))
        except (TypeError, ValueError):
            return None

    last = _f("最新") or _f("最新价") or _f("now")
    bid1 = _f("buy_1") or _f("买一")
    snapshot = {"最新价": last, "买一价": bid1, "涨停价": lp, "raw": q}

    if last is None:
        return EntrySignal(code, False, lp, "无法解析现价", snapshot)

    near_seal = last >= lp * config.entry.seal_trigger_ratio
    bidding_at_limit = bid1 is not None and abs(bid1 - lp) < 0.011
    if near_seal and (bidding_at_limit or last >= lp):
        return EntrySignal(code, True, lp,
                           f"现价{last}逼近/触及涨停{lp}，资金扫货即将封板→打板", snapshot)
    return EntrySignal(code, False, lp,
                       f"现价{last} 距涨停{lp} 未达打板临界，继续观察（不追涨）", snapshot)
