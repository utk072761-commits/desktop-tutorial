"""
首板初筛策略——全部可调参数集中在此处。

策略原文（七大“一票否决” + 强中择优 + 次日进场）逐条映射到下方阈值，
方便回测调参与实盘微调。所有金额单位为“元”，比例类字段为“百分数”（与东财一致）。
"""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class ScreenConfig:
    # ── 否决条件①：分时走势弱（反复炸板）───────────────────────────────
    # 当日炸板次数 >= 该值即剔除（主力做多不坚决 / 抛压过大）。
    # “反复”炸板取 2 次；若想更严格只保留零炸板，改为 1。
    max_zhaban_times: int = 2

    # ── 否决条件②：盘子过大 ──────────────────────────────────────────
    # 流通市值上限（元）。> 100 亿 剔除。
    max_float_mv: float = 100e8

    # ── 否决条件③：股性不活跃 ────────────────────────────────────────
    # 统计窗口（交易日，约半年）与窗口内最少涨停次数。
    activity_window_days: int = 120
    min_limit_ups_half_year: int = 2

    # ── 否决条件④：绝对价格高 ────────────────────────────────────────
    max_price: float = 20.0

    # ── 否决条件⑤：缺乏换手（一字板）─────────────────────────────────
    # 一字板判定：当日 开=高=低=收 且封于集合竞价；
    # 或 首次封板时间==09:25:00 且 换手率低于下方阈值（百分数）。
    one_word_max_turnover: float = 1.5

    # ── 否决条件⑥：位置过高（首板前已有较大涨幅）─────────────────────
    # 用“首板前一日收盘 / 窗口起点收盘 - 1”衡量前期涨幅，超过阈值剔除。
    runup_window_days: int = 60
    max_prior_gain: float = 0.50  # 50%

    # ── 否决条件⑦：单打独斗（无板块效应）─────────────────────────────
    # 当日涨停池中“同所属行业”的其它涨停股数量（不含自身）须 >= 该值。
    min_sector_peers: int = 1

    # ── 涨停幅度判定（按板块自动取 10% / 20% / 30%）的容差 ────────────
    limit_pct_tolerance: float = 0.005


@dataclass
class RankConfig:
    """强中择优（PK）权重——封板时间 / 量价 / 封板力度。"""
    w_seal_time: float = 0.35      # 封板越早越好
    w_volume_price: float = 0.30   # 拉升须有量、换手充分、非过分高开秒板
    w_seal_strength: float = 0.35  # 封板瞬间力度（封成比 / 扫单）

    # 量价配合：理想换手区间（百分数）。过低=没换手，过高=出货嫌疑。
    ideal_turnover_low: float = 5.0
    ideal_turnover_high: float = 25.0
    # 过分高开秒板惩罚：高开幅度阈值（如 > 5% 且秒板则扣分）。
    high_open_penalty_pct: float = 5.0
    # 秒板判定：首次封板时间早于该秒数（自 09:30 起）视为“秒板”。
    fast_seal_seconds: int = 60

    top_n: int = 3  # 最终锁定的自选数量（2~3 只）


@dataclass
class EntryConfig:
    """次日博弈——集合竞价观察 + 打板进场。"""
    auction_open_low: float = 3.0    # 理想高开下限（%）
    auction_open_high: float = 5.0   # 理想高开上限（%）
    # 竞价换手 / 昨日全天换手 的目标比例（约 10%）。
    auction_turnover_ratio: float = 0.10
    auction_turnover_tol: float = 0.5  # 比例容差（达到目标的 ±50% 视为合格区间）
    # 打板触发：现价 >= 涨停价 * 该系数 且买盘扫货 → 挂涨停价。
    seal_trigger_ratio: float = 0.99


@dataclass
class Config:
    screen: ScreenConfig = field(default_factory=ScreenConfig)
    rank: RankConfig = field(default_factory=RankConfig)
    entry: EntryConfig = field(default_factory=EntryConfig)
    # 网络请求失败重试（次）与退避基数（秒）。
    retries: int = 3
    backoff: float = 1.5


DEFAULT_CONFIG = Config()


def limit_pct(code: str) -> float:
    """按证券代码推断涨停幅度：创业板/科创板 20%，北交所 30%，其余主板 10%。"""
    code = str(code)
    if code.startswith(("8", "4", "920", "43")):
        return 0.30
    if code.startswith(("300", "301", "688", "689")):
        return 0.20
    return 0.10


def market_prefix(code: str) -> str:
    """生成腾讯/新浪行情所需的市场前缀（sh / sz / bj）。"""
    code = str(code)
    if code.startswith("6"):
        return "sh"
    if code.startswith(("8", "4", "920", "43")):
        return "bj"
    return "sz"
