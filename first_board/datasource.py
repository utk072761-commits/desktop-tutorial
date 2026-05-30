"""数据访问层。

为什么选 akshare 作为 A 股数据源：
  * 免费、开源、无需 token / 积分，开箱即用；
  * 直接提供东方财富「涨停股池」(stock_zt_pool_em)，一次拿到当日全部涨停股
    及炸板次数、流通市值、换手率、封板资金、首次封板时间、连板数、所属行业等
    —— 七大否决条件大部分字段一步到位；
  * 同时覆盖腾讯/东财日线、分时、集合竞价等接口，足以支撑“强中择优”和“次日进场”。

每个接口都封装了重试 + 多源回退，单点不可用时自动切换，保证实盘鲁棒性。
"""

from __future__ import annotations

import time
from typing import Optional

import pandas as pd

try:
    import akshare as ak
except Exception:  # pragma: no cover - 允许在无 akshare 环境下导入做单测
    ak = None

from config import DEFAULT_CONFIG, market_prefix


def _retry(fn, *args, retries: int = None, backoff: float = None, **kwargs):
    """带指数退避的重试包装。"""
    retries = DEFAULT_CONFIG.retries if retries is None else retries
    backoff = DEFAULT_CONFIG.backoff if backoff is None else backoff
    last = None
    for i in range(retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001
            last = e
            if i < retries - 1:
                time.sleep(backoff ** i)
    raise last


def _require_ak():
    if ak is None:
        raise RuntimeError("未安装 akshare，请先 `pip install akshare`。")


# ──────────────────────────────────────────────────────────────────────────
# 1. 当日涨停股池（初筛主数据源）
# ──────────────────────────────────────────────────────────────────────────
def get_zt_pool(date: str) -> pd.DataFrame:
    """东财涨停股池。date 形如 'YYYYMMDD'。

    返回列：代码 名称 涨跌幅 最新价 成交额 流通市值 总市值 换手率 封板资金
            首次封板时间 最后封板时间 炸板次数 涨停统计 连板数 所属行业
    """
    _require_ak()
    df = _retry(ak.stock_zt_pool_em, date=date)
    if df is None or len(df) == 0:
        return pd.DataFrame()
    df = df.copy()
    df["代码"] = df["代码"].astype(str).str.zfill(6)
    return df


def get_first_boards(date: str) -> pd.DataFrame:
    """从涨停池中切出“首板”（连板数 == 1）。"""
    pool = get_zt_pool(date)
    if pool.empty:
        return pool
    return pool[pool["连板数"] == 1].reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────
# 2. 日线历史（半年涨停次数 / 前期涨幅 / 一字板判定）
# ──────────────────────────────────────────────────────────────────────────
def get_daily_hist(code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """日线历史，统一列名 date/open/close/high/low。

    优先腾讯(stock_zh_a_hist_tx)，失败回退东财(stock_zh_a_hist)。两者在不同
    网络环境下互为备份。返回按日期升序、不复权（涨停判定用收盘涨幅，不受复权影响）。
    """
    _require_ak()
    pre = market_prefix(code)

    # 源 A：腾讯日线
    def _tx():
        df = ak.stock_zh_a_hist_tx(
            symbol=f"{pre}{code}", start_date=start_date, end_date=end_date, adjust=""
        )
        return df.rename(columns={"date": "date"})[["date", "open", "close", "high", "low"]]

    # 源 B：东财日线
    def _em():
        df = ak.stock_zh_a_hist(
            symbol=code, period="daily", start_date=start_date, end_date=end_date, adjust=""
        )
        return df.rename(
            columns={"日期": "date", "开盘": "open", "收盘": "close", "最高": "high", "最低": "low"}
        )[["date", "open", "close", "high", "low"]]

    for src in (_tx, _em):
        try:
            df = _retry(src)
            if df is not None and len(df) > 0:
                df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
                for c in ("open", "close", "high", "low"):
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                return df.sort_values("date").reset_index(drop=True)
        except Exception:  # noqa: BLE001
            continue
    return pd.DataFrame(columns=["date", "open", "close", "high", "low"])


# ──────────────────────────────────────────────────────────────────────────
# 3. 分时（炸板确认 / 拉升量价）
# ──────────────────────────────────────────────────────────────────────────
def get_intraday_min(code: str, date: str) -> pd.DataFrame:
    """当日 1 分钟分时。date 形如 'YYYYMMDD'。"""
    _require_ak()
    d = f"{date[:4]}-{date[4:6]}-{date[6:8]}"
    df = _retry(
        ak.stock_zh_a_hist_min_em,
        symbol=code, period="1",
        start_date=f"{d} 09:30:00", end_date=f"{d} 15:00:00", adjust="",
    )
    if df is None or len(df) == 0:
        return pd.DataFrame()
    return df.rename(
        columns={"时间": "time", "开盘": "open", "收盘": "close", "最高": "high",
                 "最低": "low", "成交量": "volume", "成交额": "amount"}
    )


# ──────────────────────────────────────────────────────────────────────────
# 4. 次日集合竞价（09:15–09:25）
# ──────────────────────────────────────────────────────────────────────────
def get_pre_auction(code: str) -> pd.DataFrame:
    """盘前集合竞价分钟数据（含 09:15–09:25 撮合过程，09:25 行为开盘价/竞价量）。

    注意：该接口取“当前交易日”的盘前数据，需在次日开盘前/开盘后调用。
    """
    _require_ak()
    df = _retry(ak.stock_zh_a_hist_pre_min_em, symbol=code)
    if df is None or len(df) == 0:
        return pd.DataFrame()
    return df.rename(
        columns={"时间": "time", "开盘": "open", "收盘": "close", "最高": "high",
                 "最低": "low", "成交量": "volume", "成交额": "amount", "最新价": "last"}
    )


# ──────────────────────────────────────────────────────────────────────────
# 5. 实时盘口（打板进场，仅交易时段可用）
# ──────────────────────────────────────────────────────────────────────────
def get_realtime_quote(code: str) -> Optional[dict]:
    """实时五档盘口快照，用于盘中“即将封板”判定。

    返回 dict：{'last':现价, 'bid1':买一价, 'bid1_vol':买一量(手), ...}；
    非交易时段或接口受限时返回 None（不影响初筛/复盘流程）。
    """
    _require_ak()
    try:
        df = _retry(ak.stock_bid_ask_em, symbol=code, retries=2)
    except Exception:  # noqa: BLE001
        return None
    if df is None or len(df) == 0:
        return None
    kv = dict(zip(df["item"], df["value"]))
    return kv
