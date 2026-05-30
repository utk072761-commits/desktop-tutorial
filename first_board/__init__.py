"""A股首板初筛量化系统。

模块划分：
- datasource : 数据访问层（akshare 封装 + 多源回退 + 重试）
- screening  : 七大“一票否决”初筛
- ranking    : 强中择优（封板时间 / 量价 / 封板力度 PK）
- entry      : 次日集合竞价观察 + 打板进场信号
- pipeline   : 端到端编排
"""

from .pipeline import run_pipeline, FirstBoardPipeline

__all__ = ["run_pipeline", "FirstBoardPipeline"]
