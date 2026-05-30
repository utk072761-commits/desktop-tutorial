"""核心交易准备：首板初筛——七大“一票否决”。

输入：某交易日收盘后的首板股票（连板数==1）。
输出：逐只标注是否通过、命中的否决原因，并返回通过初筛的标的。

七条规则与代码函数一一对应：
  ① 分时走势弱（反复炸板）   → _veto_zhaban
  ② 盘子过大               → _veto_float_mv
  ③ 股性不活跃             → _veto_inactive
  ④ 绝对价格高             → _veto_price
  ⑤ 缺乏换手（一字板）      → _veto_one_word
  ⑥ 位置过高（前期涨幅大）   → _veto_high_position
  ⑦ 单打独斗（无板块效应）   → _veto_lonely
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd

from config import Config, DEFAULT_CONFIG, limit_pct
from . import datasource as ds


@dataclass
class ScreenResult:
    code: str
    name: str
    passed: bool
    reasons: List[str] = field(default_factory=list)   # 命中的否决原因
    metrics: dict = field(default_factory=dict)        # 计算出的关键指标（供 PK/排查）


def _shift_date(date: str, days_back: int) -> str:
    d = dt.datetime.strptime(date, "%Y%m%d") - dt.timedelta(days=days_back)
    return d.strftime("%Y%m%d")


# ── ① 分时走势弱（反复炸板）────────────────────────────────────────────
def _veto_zhaban(row, cfg: ScreenResult, c: Config) -> Optional[str]:
    z = int(row.get("炸板次数", 0) or 0)
    cfg.metrics["炸板次数"] = z
    if z >= c.screen.max_zhaban_times:
        return f"分时反复炸板（炸板{z}次≥{c.screen.max_zhaban_times}）"
    return None


# ── ② 盘子过大 ──────────────────────────────────────────────────────────
def _veto_float_mv(row, cfg: ScreenResult, c: Config) -> Optional[str]:
    mv = float(row.get("流通市值", 0) or 0)
    cfg.metrics["流通市值"] = mv
    if mv > c.screen.max_float_mv:
        return f"流通市值过大（{mv/1e8:.1f}亿>{c.screen.max_float_mv/1e8:.0f}亿）"
    return None


# ── ④ 绝对价格高 ────────────────────────────────────────────────────────
def _veto_price(row, cfg: ScreenResult, c: Config) -> Optional[str]:
    p = float(row.get("最新价", 0) or 0)
    cfg.metrics["最新价"] = p
    if p > c.screen.max_price:
        return f"股价过高（{p:.2f}元>{c.screen.max_price:.0f}元）"
    return None


# ── ③ 股性不活跃 + ⑤ 一字板 + ⑥ 位置过高（共用日线历史）──────────────────
def _analyze_history(code: str, date: str, prev_close_pool: float, c: Config) -> dict:
    """一次拉取日线历史，计算：半年涨停次数、一字板、前期涨幅。"""
    start = _shift_date(date, 400)  # 拉够一年自然日 ≈ 半年交易日 + 涨幅窗口
    end = date
    hist = ds.get_daily_hist(code, start, end)
    out = {"limit_ups": None, "one_word": None, "prior_gain": None, "hist_rows": len(hist)}
    if hist.empty or len(hist) < 2:
        return out

    lp = limit_pct(code)
    thr = lp - c.screen.limit_pct_tolerance
    pct = hist["close"].pct_change()

    # ③ 近半年涨停次数（窗口内收盘涨幅≥涨停阈值的天数，含当日板）
    win = hist.tail(c.screen.activity_window_days)
    out["limit_ups"] = int((win["close"].pct_change() >= thr).sum())

    # ⑤ 一字板：当日（末行）开=高=低=收 且收盘涨幅达涨停
    last = hist.iloc[-1]
    o, h, l, cl = last["open"], last["high"], last["low"], last["close"]
    day_pct = pct.iloc[-1] if len(pct) else 0
    out["one_word"] = bool(o == h == l == cl and day_pct >= thr)

    # ⑥ 前期涨幅：首板前一日收盘 / 窗口起点收盘 - 1
    w = c.screen.runup_window_days
    if len(hist) >= w + 2:
        prev_close = hist["close"].iloc[-2]          # 首板前一日
        base_close = hist["close"].iloc[-(w + 2)]    # 窗口起点
        if base_close:
            out["prior_gain"] = float(prev_close / base_close - 1)
    return out


def _veto_inactive(hist_info: dict, cfg: ScreenResult, c: Config) -> Optional[str]:
    n = hist_info.get("limit_ups")
    cfg.metrics["半年涨停次数"] = n
    if n is None:
        return None  # 无历史数据，无法判定→不否决（保守放行，交由人工核查）
    if n < c.screen.min_limit_ups_half_year:
        return f"股性不活跃（半年涨停{n}次<{c.screen.min_limit_ups_half_year}次）"
    return None


def _veto_one_word(row, hist_info: dict, cfg: ScreenResult, c: Config) -> Optional[str]:
    one_word = hist_info.get("one_word")
    # 历史源不可用时，用涨停池字段兜底：09:25:00 封板 且 换手极低
    seal_time = str(row.get("首次封板时间", "") or "")
    turnover = float(row.get("换手率", 0) or 0)
    fallback = seal_time == "092500" and turnover < c.screen.one_word_max_turnover
    cfg.metrics["一字板"] = bool(one_word) or fallback
    if one_word or fallback:
        return "一字板封死（开盘无换手，承接力无法判断）"
    return None


def _veto_high_position(hist_info: dict, cfg: ScreenResult, c: Config) -> Optional[str]:
    g = hist_info.get("prior_gain")
    cfg.metrics["前期涨幅"] = g
    if g is None:
        return None
    if g > c.screen.max_prior_gain:
        return f"位置过高（首板前{c.screen.runup_window_days}日已涨{g*100:.0f}%>{c.screen.max_prior_gain*100:.0f}%）"
    return None


# ── ⑦ 单打独斗（无板块效应）────────────────────────────────────────────
def _veto_lonely(row, pool: pd.DataFrame, cfg: ScreenResult, c: Config) -> Optional[str]:
    """同一“所属行业”在当日**整个涨停池**中的其它涨停股数量（不含自身）。"""
    industry = row.get("所属行业", None)
    if not industry or "所属行业" not in pool.columns:
        return None
    peers = pool[(pool["所属行业"] == industry) & (pool["代码"] != row["代码"])]
    n = len(peers)
    cfg.metrics["同板块涨停数"] = n
    cfg.metrics["所属行业"] = industry
    if n < c.screen.min_sector_peers:
        return f"单打独斗（同板块『{industry}』另有涨停{n}只<{c.screen.min_sector_peers}只）"
    return None


# ── 主流程 ──────────────────────────────────────────────────────────────
def screen_first_boards(
    date: str,
    pool: Optional[pd.DataFrame] = None,
    config: Config = DEFAULT_CONFIG,
    use_history: bool = True,
) -> List[ScreenResult]:
    """对某交易日的首板执行七大否决初筛。

    date        : 'YYYYMMDD'（首板当日）。
    pool        : 可传入完整涨停池（用于板块效应统计）；缺省自动拉取。
    use_history : 是否拉取日线历史以判定③⑤⑥（实盘建议 True；快速演示可 False）。
    """
    if pool is None:
        pool = ds.get_zt_pool(date)
    if pool.empty:
        return []
    first = pool[pool["连板数"] == 1].reset_index(drop=True)

    results: List[ScreenResult] = []
    for _, row in first.iterrows():
        r = ScreenResult(code=row["代码"], name=row.get("名称", ""), passed=True)

        # 不依赖历史的快速否决（②④①⑦）
        for fn in (_veto_zhaban, _veto_float_mv, _veto_price):
            reason = fn(row, r, config)
            if reason:
                r.reasons.append(reason)

        lonely = _veto_lonely(row, pool, r, config)
        if lonely:
            r.reasons.append(lonely)

        # 依赖历史的否决（③⑤⑥）
        if use_history:
            try:
                info = _analyze_history(row["代码"], date, row.get("最新价"), config)
            except Exception as e:  # noqa: BLE001
                info = {"limit_ups": None, "one_word": None, "prior_gain": None,
                        "error": str(e)}
            for fn in (_veto_inactive, _veto_high_position):
                reason = fn(info, r, config)
                if reason:
                    r.reasons.append(reason)
            ow = _veto_one_word(row, info, r, config)
            if ow:
                r.reasons.append(ow)
        else:
            # 仅用涨停池字段兜底判一字板
            ow = _veto_one_word(row, {}, r, config)
            if ow:
                r.reasons.append(ow)

        # 记录 PK 阶段要用到的原始字段
        r.metrics.update({
            "首次封板时间": str(row.get("首次封板时间", "")),
            "换手率": float(row.get("换手率", 0) or 0),
            "封板资金": float(row.get("封板资金", 0) or 0),
            "涨跌幅": float(row.get("涨跌幅", 0) or 0),
        })
        r.passed = len(r.reasons) == 0
        results.append(r)
    return results
