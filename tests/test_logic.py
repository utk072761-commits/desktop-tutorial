"""纯逻辑单测（不依赖网络）：验证否决规则与打分函数的判定。"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from config import DEFAULT_CONFIG, limit_pct, market_prefix
from first_board.screening import (
    _veto_zhaban, _veto_float_mv, _veto_price, _veto_lonely,
    _veto_inactive, _veto_high_position, _veto_one_word, ScreenResult,
)
from first_board.ranking import _score_seal_time, _seal_seconds, _score_seal_strength
from first_board.entry import round_limit_price


def _blank():
    return ScreenResult(code="000001", name="测试", passed=True)


def test_limit_pct_and_prefix():
    assert limit_pct("600000") == 0.10
    assert limit_pct("300750") == 0.20
    assert limit_pct("688981") == 0.20
    assert limit_pct("830799") == 0.30
    assert market_prefix("600000") == "sh"
    assert market_prefix("000001") == "sz"
    assert market_prefix("830799") == "bj"


def test_veto_zhaban():
    assert _veto_zhaban({"炸板次数": 2}, _blank(), DEFAULT_CONFIG)  # 命中
    assert _veto_zhaban({"炸板次数": 1}, _blank(), DEFAULT_CONFIG) is None


def test_veto_float_mv():
    assert _veto_float_mv({"流通市值": 150e8}, _blank(), DEFAULT_CONFIG)
    assert _veto_float_mv({"流通市值": 50e8}, _blank(), DEFAULT_CONFIG) is None


def test_veto_price():
    assert _veto_price({"最新价": 25.0}, _blank(), DEFAULT_CONFIG)
    assert _veto_price({"最新价": 12.0}, _blank(), DEFAULT_CONFIG) is None


def test_veto_lonely():
    pool = pd.DataFrame({
        "代码": ["000001", "000002", "000003"],
        "所属行业": ["银行", "银行", "钢铁"],
    })
    # 000001 同行业银行另有 000002 → 不孤立
    assert _veto_lonely({"代码": "000001", "所属行业": "银行"}, pool, _blank(), DEFAULT_CONFIG) is None
    # 000003 钢铁仅自身 → 孤立
    assert _veto_lonely({"代码": "000003", "所属行业": "钢铁"}, pool, _blank(), DEFAULT_CONFIG)


def test_veto_inactive():
    assert _veto_inactive({"limit_ups": 1}, _blank(), DEFAULT_CONFIG)
    assert _veto_inactive({"limit_ups": 3}, _blank(), DEFAULT_CONFIG) is None
    assert _veto_inactive({"limit_ups": None}, _blank(), DEFAULT_CONFIG) is None  # 无数据放行


def test_veto_high_position():
    assert _veto_high_position({"prior_gain": 0.8}, _blank(), DEFAULT_CONFIG)
    assert _veto_high_position({"prior_gain": 0.2}, _blank(), DEFAULT_CONFIG) is None


def test_veto_one_word():
    # 历史确认一字板
    assert _veto_one_word({"换手率": 5}, {"one_word": True}, _blank(), DEFAULT_CONFIG)
    # 池字段兜底：09:25 封板 + 极低换手
    row = {"首次封板时间": "092500", "换手率": 0.5}
    assert _veto_one_word(row, {}, _blank(), DEFAULT_CONFIG)
    # 有换手 → 非一字
    row2 = {"首次封板时间": "093500", "换手率": 8.0}
    assert _veto_one_word(row2, {"one_word": False}, _blank(), DEFAULT_CONFIG) is None


def test_seal_time_scoring():
    assert _seal_seconds("093000") == 0
    assert _seal_seconds("103000") == 3600
    # 早封分高于晚封
    assert _score_seal_time("093000") > _score_seal_time("143000")


def test_seal_strength_monotonic():
    weak = _score_seal_strength(seal_fund=1e6, float_mv=50e8, burst=None)
    strong = _score_seal_strength(seal_fund=3e8, float_mv=50e8, burst=None)
    assert strong > weak


def test_round_limit_price():
    assert round_limit_price(10.0, "600000") == 11.0   # 10% 板
    assert round_limit_price(10.0, "300750") == 12.0   # 20% 板


if __name__ == "__main__":
    import subprocess
    raise SystemExit(subprocess.call(["pytest", "-q", __file__]))
