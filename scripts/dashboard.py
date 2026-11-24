"""训练过程可视化网页（标准库 http.server，零第三方依赖）。

启动：python -I scripts/dashboard.py [--port 8000]
页面：http://127.0.0.1:8000/

接口：
  GET  /api/runs                列出 runs/ 下所有实验
  GET  /api/metrics?run=NAME    读取该实验全部 metrics.jsonl
  GET  /api/state?run=NAME      读取最新 state.json
  POST /api/control?run=NAME    写 control.json（固定/解冻 scale、手动指定值）
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_DIR = os.path.join(ROOT, "runs")
WEB_DIR = os.path.join(ROOT, "web")


def _atomic_write_json(path: str, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path)


def list_runs():
    if not os.path.isdir(RUNS_DIR):
        return []
    out = []
    for name in sorted(os.listdir(RUNS_DIR)):
        d = os.path.join(RUNS_DIR, name)
        if not os.path.isdir(d):
            continue
        m = os.path.join(d, "metrics.jsonl")
        if os.path.exists(m):
            out.append({
                "name": name,
                "steps": sum(1 for _ in open(m, encoding="utf-8")),
                "has_state": os.path.exists(os.path.join(d, "state.json")),
            })
    return out


def read_metrics(run: str):
    # 防路径穿越
    safe = os.path.basename(run)
    path = os.path.join(RUNS_DIR, safe, "metrics.jsonl")
    if not os.path.exists(path):
        return None
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_state(run: str):
    safe = os.path.basename(run)
    path = os.path.join(RUNS_DIR, safe, "state.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_control(run: str, payload: dict):
    safe = os.path.basename(run)
    d = os.path.join(RUNS_DIR, safe)
    os.makedirs(d, exist_ok=True)
    fixed = bool(payload.get("fixed", False))
    scale = payload.get("scale")
    if scale is not None:
        scale = float(scale)
        if not (scale > 0):
            raise ValueError("scale 必须为正数")
    _atomic_write_json(os.path.join(d, "control.json"),
                       {"fixed": fixed, "scale": scale})
    return {"ok": True, "fixed": fixed, "scale": scale}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # 安静一点
        pass

    def _send(self, code, body: bytes, ctype="application/json; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"))

    def do_GET(self):
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)
        if parsed.path == "/api/runs":
            return self._json({"runs": list_runs()})
        if parsed.path == "/api/metrics":
            run = (qs.get("run") or [""])[0]
            rows = read_metrics(run)
            if rows is None:
                return self._json({"error": f"run {run!r} 不存在"}, 404)
            return self._json({"metrics": rows})
        if parsed.path == "/api/state":
            run = (qs.get("run") or [""])[0]
            state = read_state(run)
            if state is None:
                return self._json({"error": "state 不存在"}, 404)
            return self._json(state)
        # 静态文件
        if parsed.path in ("/", "/index.html"):
            static = "index.html"
        else:
            static = parsed.path.lstrip("/")
        fpath = os.path.normpath(os.path.join(WEB_DIR, static))
        if not fpath.startswith(WEB_DIR) or not os.path.isfile(fpath):
            return self._send(404, b"not found", "text/plain; charset=utf-8")
        ctype = ("text/html; charset=utf-8" if fpath.endswith(".html") else
                 "application/javascript; charset=utf-8")
        with open(fpath, "rb") as f:
            self._send(200, f.read(), ctype)

    def do_POST(self):
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)
        if parsed.path != "/api/control":
            return self._json({"error": "未知接口"}, 404)
        run = (qs.get("run") or [""])[0]
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            result = write_control(run, payload)
        except (ValueError, json.JSONDecodeError) as exc:
            return self._json({"error": str(exc)}, 400)
        self._json(result)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"看板已启动: http://{args.host}:{args.port}/  (Ctrl+C 退出)")
    print(f"实验目录: {RUNS_DIR}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已退出")
        sys.exit(0)


if __name__ == "__main__":
    main()
