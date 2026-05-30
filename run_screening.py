#!/usr/bin/env python3
"""命令行入口：A 股首板初筛 + 强中择优 + 次日集合竞价观察。

用法：
    # 复盘某交易日首板，输出自选池
    python run_screening.py --date 20260529

    # 快速模式（不拉日线历史，跳过③⑤⑥的历史判定，速度快）
    python run_screening.py --date 20260529 --no-history

    # 顺带对自选池做“次日集合竞价”检查（需在次日盘前/盘中运行）
    python run_screening.py --date 20260529 --auction
"""

import argparse
import datetime as dt

from config import DEFAULT_CONFIG
from first_board.pipeline import FirstBoardPipeline


def main():
    ap = argparse.ArgumentParser(description="A股首板初筛量化系统")
    ap.add_argument("--date", default=dt.date.today().strftime("%Y%m%d"),
                    help="首板当日 YYYYMMDD（默认今天）")
    ap.add_argument("--no-history", action="store_true",
                    help="不拉日线历史（跳过半年涨停数/前期涨幅/一字板的历史判定）")
    ap.add_argument("--no-burst", action="store_true",
                    help="不做分时放量尖峰二次确认（更快）")
    ap.add_argument("--auction", action="store_true",
                    help="对自选池执行次日集合竞价检查")
    ap.add_argument("--top", type=int, default=None, help="覆盖自选数量 top_n")
    args = ap.parse_args()

    cfg = DEFAULT_CONFIG
    if args.top:
        cfg.rank.top_n = args.top

    pipe = FirstBoardPipeline(cfg)
    out = pipe.run(args.date, use_history=not args.no_history,
                   confirm_burst=not args.no_burst)
    print(out.summary())

    if args.auction and out.watchlist:
        print("\n── 次日集合竞价观察 ──")
        for chk in pipe.next_day_auction(out.watchlist, args.date):
            tag = "✓符合" if chk.qualified else "✗不符"
            op = f"{chk.open_pct:.1f}%" if chk.open_pct is not None else "NA"
            print(f"  {chk.code} {chk.name} [{tag}] 高开{op} "
                  f"竞价换手比={chk.turnover_ratio}  {('; '.join(chk.notes)) or ''}")
        print("\n进场提醒：仅在资金扫货、即将封死涨停的瞬间挂涨停价打板；"
              "先涨停、先符合条件，就打谁。切勿盘中追涨。")


if __name__ == "__main__":
    main()
