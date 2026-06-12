"""决策工作台后端(蓝图 §7)——零依赖 stdlib http.server。

把 Roundtable 编排器接到三区 UI,并与后端状态机绑定:
  - 圆桌视窗(Debate Floor):messages 按 round 分组、显示 refs 连线(§7.1)
  - 决策摘要板(Decision Board):渲染 DisputeMatrix 三栏(§7.2)
  - 行动控制台(Action Panel):介入/通过/拒绝/中断;通过是唯一触发副作用的边(§7.3)

默认用 MockAdapter,故 `python -m roundtable.ui.server` 即可在浏览器里玩通全流程,
无需任何 API key。接真实模型时把 build_registry 换成 providers.* 即可。
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, Optional

from ..adapters import MockAdapter
from ..session import Roundtable
from ..state import State
from ..store import session_to_dict
from ..tools import Tool

_HERE = os.path.dirname(os.path.abspath(__file__))


def build_registry(use_tools: bool = False, latency: float = 0.0) -> dict:
    """演示用三位舍人;A2/A3 第 2 轮起让步以演示收敛。

    use_tools 时 A1 会在发言前先调一次 search 工具(其轨迹会在 UI 圆桌视窗呈现)。
    latency 为每位发言的总耗时,均摊到逐 token,供 UI 流式渲染可见。
    """
    a1_extra = {"tool_call": "search", "tool_args": {"q": "微服务运维成本"}} if use_tools else {}
    return {
        "A1": MockAdapter("A1", stance_seed="应采用单体架构，快速交付",
                          arguments=["团队规模小，微服务运维成本高", "需求未稳定，边界易变"],
                          latency=latency, **a1_extra),
        "A2": MockAdapter("A2", stance_seed="应采用微服务，长期可扩展",
                          arguments=["未来流量增长确定", "团队需独立部署"],
                          concede_at=2, concede_to="A1", latency=latency),
        "A3": MockAdapter("A3", stance_seed="折中：模块化单体起步",
                          arguments=["保留拆分边界", "先验证再投入运维"],
                          concede_at=2, concede_to="A1", latency=latency),
    }


def demo_tools() -> list:
    """演示工具:返回确定性的运维成本基准数据。"""
    def search(args):
        return f"基准数据：{args.get('q', '')} ≈ 3 人/月"

    return [Tool(
        name="search",
        description="检索运维成本基准数据",
        input_schema={"type": "object", "properties": {"q": {"type": "string"}},
                      "required": ["q"]},
        handler=search,
    )]


class RoundtableServer:
    """持有一个活跃 Roundtable 会话,供 HTTP handler 驱动。

    background=True 时辩论跑在后台线程,/api/state 实时暴露:
      - busy: 辩论是否进行中(前端据此轮询);
      - live: {agent_id: 累积中的部分发言}(逐 token 流式缓冲,每轮结束清空);
    「介入辩论」走 run_debate 的 interrupt 探针 -> DISCUSSION --user_interrupt--> PROPOSAL。
    读写竞争为演示级容忍:state 读快照,live 加锁。
    """

    def __init__(self) -> None:
        self.rt: Optional[Roundtable] = None
        self.committed: list[str] = []
        self.busy = False
        self._interrupt = False
        self._live: dict[str, str] = {}
        self._lock = threading.Lock()

    def state_payload(self) -> dict:
        with self._lock:
            live = dict(self._live)
        if self.rt is None:
            return {"state": "WAITING", "session": None, "busy": False, "live": {}}
        return {
            "state": self.rt.session.state.value,
            "session": session_to_dict(self.rt.session),
            "committed": self.committed,
            "busy": self.busy,
            "live": live,
        }

    def start(
        self,
        prompt: str,
        debate: bool,
        n_max: int,
        use_tools: bool = False,
        background: bool = False,
        latency: Optional[float] = None,
    ) -> dict:
        if self.busy:
            return self.state_payload()  # 已有辩论进行中,忽略重复开始
        # 后台(浏览器)模式给 Mock 加延迟,让逐 token 流式可见;同步(测试)模式保持 0。
        if latency is None:
            latency = 1.0 if background else 0.0
        self.rt = Roundtable("ui-session", build_registry(use_tools, latency=latency))
        self.committed = []
        self._interrupt = False
        with self._lock:
            self._live = {}
        tools = demo_tools() if use_tools else None

        def on_token(aid: str, tok: str) -> None:
            with self._lock:
                self._live[aid] = self._live.get(aid, "") + tok

        def on_round(n: int) -> None:
            # 本轮发言已落定进 session.messages,清空流式缓冲避免重复显示。
            with self._lock:
                self._live = {}

        def runner() -> None:
            try:
                if debate:
                    asyncio.run(self.rt.run_debate(
                        prompt, n_max=n_max, tools=tools, on_token=on_token,
                        on_round=on_round, interrupt=lambda: self._interrupt))
                else:
                    asyncio.run(self.rt.ask_single(prompt, tools=tools, on_token=on_token))
            finally:
                self.busy = False
                with self._lock:
                    self._live = {}

        if background:
            self.busy = True
            threading.Thread(target=runner, daemon=True).start()
        else:
            runner()
        return self.state_payload()

    def interrupt(self) -> dict:
        """「介入辩论」:置中断旗标,辩论在当前轮结束后截断进 PROPOSAL(§1/§7.3)。"""
        if self.busy:
            self._interrupt = True
        return self.state_payload()

    def approve(self) -> dict:
        if self.rt and self.rt.session.state == State.PROPOSAL:
            # 副作用只能挂在 COMMITTED 之后(§7.3)。
            self.rt.approve(side_effect=lambda: self.committed.append("决策已落地（示意写入）"))
        return self.state_payload()

    def reject(self, reopen: bool, new_instruction: bool) -> dict:
        if self.rt and self.rt.session.state == State.PROPOSAL:
            self.rt.reject(reopen=reopen, new_instruction=new_instruction)
        return self.state_payload()

    def abort(self) -> dict:
        if self.busy:
            # 后台线程持有状态机,直接 fire 会与轮转换竞争;
            # 改走中断探针,在轮边界优雅截断(随后用户可在 PROPOSAL 拒绝/重开)。
            return self.interrupt()
        if self.rt:
            self.rt.abort()
        return self.state_payload()


def _make_handler(app: RoundtableServer):
    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, obj: dict, code: int = 200) -> None:
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> dict:
            length = int(self.headers.get("Content-Length", "0"))
            if not length:
                return {}
            return json.loads(self.rfile.read(length) or b"{}")

        def do_GET(self):  # noqa: N802
            if self.path in ("/", "/index.html"):
                with open(os.path.join(_HERE, "index.html"), "rb") as f:
                    body = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/api/state":
                self._send_json(app.state_payload())
            else:
                self.send_error(404)

        def do_POST(self):  # noqa: N802
            data = self._read_json()
            if self.path == "/api/start":
                self._send_json(app.start(
                    data.get("prompt", ""), bool(data.get("debate", True)),
                    int(data.get("n_max", 5)), bool(data.get("use_tools", False)),
                    background=bool(data.get("background", False))))
            elif self.path == "/api/interrupt":
                self._send_json(app.interrupt())
            elif self.path == "/api/approve":
                self._send_json(app.approve())
            elif self.path == "/api/reject":
                self._send_json(app.reject(
                    bool(data.get("reopen", False)), bool(data.get("new_instruction", False))))
            elif self.path == "/api/abort":
                self._send_json(app.abort())
            else:
                self.send_error(404)

        def log_message(self, *args):  # 静音默认访问日志
            pass

    return Handler


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    app = RoundtableServer()
    httpd = ThreadingHTTPServer((host, port), _make_handler(app))
    print(f"决策工作台已启动: http://{host}:{port}  (Ctrl-C 退出)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n已退出。")
        httpd.server_close()


if __name__ == "__main__":
    serve()
