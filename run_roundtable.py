"""中书省圆桌会议——端到端演示（无需任何 API key）。

用 MockAdapter 组三位舍人（A1/A2/A3）+ 外置裁判，跑通：
  单问模式  ->  辩论模式收敛  ->  分歧矩阵  ->  用户终审(画敕/封驳)

运行： python run_roundtable.py
"""

import asyncio

from roundtable import MockAdapter, Roundtable, Tool
from roundtable.decision import render_board
from roundtable.state import State


def build_registry() -> dict:
    """三位中书舍人，立场各异；A2/A3 在后续轮次让步以模拟收敛。"""
    return {
        "A1": MockAdapter("A1", stance_seed="应采用单体架构,快速交付",
                          arguments=["团队规模小,微服务运维成本高", "需求未稳定,边界易变"]),
        "A2": MockAdapter("A2", stance_seed="应采用微服务,长期可扩展",
                          arguments=["未来流量增长确定", "团队需独立部署"],
                          concede_at=2, concede_to="A1"),
        "A3": MockAdapter("A3", stance_seed="折中:模块化单体起步",
                          arguments=["保留拆分边界", "先验证再投入运维"],
                          concede_at=2, concede_to="A1"),
    }


async def main() -> None:
    rt = Roundtable("demo-session", build_registry())

    print("########## 单问模式（@A1 直接回答，不进辩论流）##########")
    answers = await rt.ask_single("@A1 这个项目该用什么架构？")
    for m in answers:
        print(f"  [{m.agent_id}] {m.content.splitlines()[0]}")
    print(f"  状态机回到: {rt.session.state.value}\n")

    print("########## 辩论模式（全员，最多 5 轮，外置裁判判收敛）##########")
    matrix = await rt.run_debate("架构选型：单体 vs 微服务？大家辩。", n_max=5)
    print(f"  辩论轮数: {rt.session.round_counter}")
    v = rt.session.verdict
    print(f"  裁判裁决: converged={v.converged}, confidence={v.confidence}")
    print(f"  token 账本: {rt.session.token_ledger}")
    print()
    print(render_board(matrix))
    print(f"\n  当前状态(应为 PROPOSAL): {rt.session.state.value}")

    print("\n########## tool_use：模型先调工具再作答（agentic 循环）##########")
    calls = {"n": 0}

    def search(args):
        calls["n"] += 1
        return f"运维成本基准：{args.get('q', '')} 约为 3 人/月"

    tool = Tool(
        name="search",
        description="检索运维成本基准数据",
        input_schema={"type": "object", "properties": {"q": {"type": "string"}},
                      "required": ["q"]},
        handler=search,
    )
    rt_tool = Roundtable("demo-tools", {
        "A1": MockAdapter("A1", stance_seed="先查数据再下结论",
                          tool_call="search", tool_args={"q": "微服务运维"}),
    })
    answers = await rt_tool.ask_single("微服务的运维成本到底多高？", tools=[tool])
    print(f"  工具被调用 {calls['n']} 次")
    print(f"  [A1] {answers[0].content.splitlines()[0]}")

    print("\n########## 终审：副作用只能挂在 COMMITTED 之后（§7.3）##########")

    def commit_side_effect():
        # 这里才允许写文件 / 发请求 / 下单——状态已合法跨过 PROPOSAL→COMMITTED。
        return "✅ 外部操作已执行（示意：写入决策记录）"

    result = rt.approve(side_effect=commit_side_effect)
    print(f"  {result}")
    print(f"  终审后状态(应为 WAITING): {rt.session.state.value}")
    assert rt.session.state == State.WAITING


if __name__ == "__main__":
    asyncio.run(main())
