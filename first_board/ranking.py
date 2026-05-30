"""核心交易准备：强中择优——锁定 2~3 只自选。

对通过七步初筛的标的，从三个维度打分 PK：
  1) 封板时间：越早越好（早盘最佳）。
  2) 量价配合：拉升须有量、换手充分；且非“过分高开后的秒板”。
  3) 封板力度：盯封板瞬间的扫单买量（资金做多决心）。

说明（关于“封板瞬间扫单”）：
  真·逐笔“瞬间扫单买量”需要 Level-2 / 逐笔成交(tick) 数据。免费源只提供
  “封板资金”（静态封单金额）。这里以「封成比 = 封板资金 / 流通市值」作为
  力度代理指标，并在 detect_seal_burst() 中用分时成交量在首封时刻的放量
  尖峰做二次确认；若接入逐笔数据，可在 seal_burst_from_tick() 中替换为
  真实扫单量，权重不变即可平滑升级。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd

from config import Config, DEFAULT_CONFIG
from . import datasource as ds
from .screening import ScreenResult


@dataclass
class RankedPick:
    code: str
    name: str
    score: float
    seal_time_score: float
    volume_price_score: float
    seal_strength_score: float
    detail: dict


def _seal_seconds(seal_time: str) -> float:
    """首次封板时间 HHMMSS → 自 09:30 起的秒数；非法值给一个很大的数（最差）。"""
    s = str(seal_time).zfill(6)
    try:
        hh, mm, ss = int(s[:2]), int(s[2:4]), int(s[4:6])
    except ValueError:
        return 6 * 3600
    secs = (hh - 9) * 3600 + (mm - 30) * 60 + ss
    return float(max(secs, 0))


def _norm(value: float, lo: float, hi: float) -> float:
    """线性归一到 [0,1]，越大越好。"""
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


# ── 维度1：封板时间 ──────────────────────────────────────────────────────
def _score_seal_time(seal_time: str) -> float:
    """09:30 封板=1 分；14:57 封板≈0 分（线性，越早越高）。"""
    secs = _seal_seconds(seal_time)
    full_day = (15 - 9) * 3600 - 30 * 60  # 09:30→15:00
    return 1.0 - _norm(secs, 0, full_day)


# ── 维度2：量价配合 ──────────────────────────────────────────────────────
def _score_volume_price(turnover: float, high_open_pct: float, seal_time: str,
                        c: Config) -> float:
    """换手在理想区间得满分；过分高开+秒板扣分。"""
    rc = c.rank
    # 换手率落在 [low, high] 给 1 分，越偏离越低
    if rc.ideal_turnover_low <= turnover <= rc.ideal_turnover_high:
        t_score = 1.0
    elif turnover < rc.ideal_turnover_low:
        t_score = _norm(turnover, 0, rc.ideal_turnover_low)            # 换手不足
    else:
        t_score = 1.0 - _norm(turnover, rc.ideal_turnover_high,
                              rc.ideal_turnover_high * 2)              # 换手过度
    # 过分高开后的秒板：高开>阈值 且 秒板 → 惩罚
    penalty = 0.0
    if high_open_pct is not None and high_open_pct > rc.high_open_penalty_pct \
            and _seal_seconds(seal_time) <= rc.fast_seal_seconds:
        penalty = 0.4
    return max(0.0, t_score - penalty)


# ── 维度3：封板力度 ──────────────────────────────────────────────────────
def _score_seal_strength(seal_fund: float, float_mv: float, burst: float) -> float:
    """封成比（封板资金/流通市值）为主，分时放量尖峰 burst∈[0,1] 加权。"""
    ratio = seal_fund / float_mv if float_mv else 0.0
    # 封成比 5% 以上视为很强（短线首板封单占流通 5% 已属大单）
    base = _norm(ratio, 0.0, 0.05)
    return 0.7 * base + 0.3 * (burst if burst is not None else base)


def detect_seal_burst(code: str, date: str, seal_time: str) -> float:
    """用 1 分钟分时确认首封时刻是否放量尖峰：返回 [0,1]。

    定义：首封所在分钟成交量 / 当日分钟成交量均值，归一到 [0,1]
    （>3 倍均量记满分）。分时不可用时返回 None，由封成比兜底。
    """
    try:
        mins = ds.get_intraday_min(code, date)
    except Exception:  # noqa: BLE001
        return None
    if mins is None or mins.empty:
        return None
    s = str(seal_time).zfill(6)
    hhmm = f"{s[:2]}:{s[2:4]}"
    mins["hm"] = mins["time"].astype(str).str[11:16]
    vol = pd.to_numeric(mins["volume"], errors="coerce")
    avg = vol[vol > 0].mean()
    hit = mins[mins["hm"] == hhmm]
    if hit.empty or not avg:
        return None
    seal_vol = pd.to_numeric(hit["volume"], errors="coerce").iloc[0]
    return _norm(seal_vol / avg, 1.0, 3.0)


def seal_burst_from_tick(tick_df: pd.DataFrame) -> float:  # pragma: no cover - 升级接口
    """【可选升级】接入逐笔成交后，统计封板瞬间最大单笔扫单买量并归一。

    tick_df 需含列：成交价/成交量/买卖方向。此处留作接口占位，逻辑不变。
    """
    raise NotImplementedError("接入 Level-2 / 逐笔数据后实现真·瞬间扫单统计。")


def rank_picks(
    date: str,
    passed: List[ScreenResult],
    config: Config = DEFAULT_CONFIG,
    confirm_burst: bool = True,
) -> List[RankedPick]:
    """对通过初筛的标的 PK 打分，返回按总分降序的前 top_n 只自选。"""
    rc = config.rank
    picks: List[RankedPick] = []
    for r in passed:
        m = r.metrics
        seal_time = m.get("首次封板时间", "")
        turnover = m.get("换手率", 0.0)
        seal_fund = m.get("封板资金", 0.0)
        float_mv = m.get("流通市值", 0.0)
        # 高开幅度：收盘涨幅近似不可得开盘，用分时首分钟开盘价/前收估算
        high_open_pct = _estimate_high_open(r.code, date)

        burst = detect_seal_burst(r.code, date, seal_time) if confirm_burst else None

        s_time = _score_seal_time(seal_time)
        s_vp = _score_volume_price(turnover, high_open_pct, seal_time, config)
        s_strength = _score_seal_strength(seal_fund, float_mv, burst)
        total = (rc.w_seal_time * s_time
                 + rc.w_volume_price * s_vp
                 + rc.w_seal_strength * s_strength)

        picks.append(RankedPick(
            code=r.code, name=r.name, score=round(total, 4),
            seal_time_score=round(s_time, 3),
            volume_price_score=round(s_vp, 3),
            seal_strength_score=round(s_strength, 3),
            detail={
                "首次封板时间": seal_time, "换手率": turnover,
                "封板资金": seal_fund, "封成比": round(seal_fund / float_mv, 4) if float_mv else None,
                "高开幅度%": high_open_pct, "放量尖峰": burst,
            },
        ))
    picks.sort(key=lambda p: p.score, reverse=True)
    return picks[: rc.top_n]


def _estimate_high_open(code: str, date: str) -> float:
    """高开幅度(%) = 首板当日开盘价 / 前一交易日收盘价 - 1，失败返回 None。"""
    import datetime as _dt
    try:
        start = (_dt.datetime.strptime(date, "%Y%m%d") - _dt.timedelta(days=20)
                 ).strftime("%Y%m%d")
        hist = ds.get_daily_hist(code, start, date)
        if hist is None or len(hist) < 2:
            return None
        open_today = hist["open"].iloc[-1]
        prev_close = hist["close"].iloc[-2]
        if not prev_close:
            return None
        return float((open_today / prev_close - 1) * 100)
    except Exception:  # noqa: BLE001
        return None
