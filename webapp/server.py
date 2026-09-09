"""
WaferLens OS -- the local mission-control server.

    python -m webapp.server            # http://127.0.0.1:8000
    python -m webapp.server --port 9000 --dataset smoke

Standard library only. The project already asks that no dependency be added without a
line in requirements-ours.txt, and a local demo console does not need a framework.
Artifacts load once at start; every request is served from memory apart from the block
waveforms, which are read from the memmap on demand.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from . import data

STATIC = Path(__file__).resolve().parent / "static"
FIGURES = Path(__file__).resolve().parents[1] / "figures"


def _num(qs: dict, key: str, default: float) -> float:
    try:
        return float(qs.get(key, [default])[0])
    except (TypeError, ValueError):
        return default


class Handler(BaseHTTPRequestHandler):
    server_version = "WaferLensOS/1.0"
    protocol_version = "HTTP/1.1"

    # ---------------------------------------------------------------- plumbing
    def log_message(self, fmt, *args):        # keep the console readable
        if "--verbose" in getattr(self.server, "flags", []):
            super().log_message(fmt, *args)

    def _send(self, code: int, body: bytes, ctype: str, cache: str = "no-store"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, payload, code: int = 200):
        body = json.dumps(payload, allow_nan=False).encode("utf-8")
        self._send(code, body, "application/json; charset=utf-8")

    def _error(self, code: int, message: str, detail: str | None = None):
        self._json({"error": message, "detail": detail}, code)

    # ---------------------------------------------------------------- routing
    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        url = urlparse(self.path)
        route = unquote(url.path)
        qs = parse_qs(url.query)
        try:
            if route.startswith("/api/"):
                return self._api(route[5:], qs)
            return self._static(route)
        except data.MissingArtifact as exc:
            self._error(404, str(exc))
        except Exception as exc:                                  # noqa: BLE001
            traceback.print_exc()
            self._error(500, f"{type(exc).__name__}: {exc}")

    def _api(self, name: str, qs: dict):
        parts = [p for p in name.split("/") if p]
        head = parts[0] if parts else ""

        if head == "summary":
            return self._json(data.summary())
        if head == "wafers":
            return self._json(data.wafer_list())
        if head == "wafer" and len(parts) >= 2:
            return self._json(data.wafer(parts[1]))
        if head == "die" and len(parts) >= 4:
            return self._json(data.die(parts[1], int(parts[2]), int(parts[3]),
                                       with_blocks=qs.get("blocks", ["1"])[0] != "0"))
        if head == "benchmarks":
            return self._json(data.benchmarks())
        if head == "imbalance":
            return self._json(data.imbalance())
        if head == "calibration":
            return self._json(data.calibration())
        if head == "economics":
            return self._json(data.economics(
                cost_ratio=_num(qs, "cost_ratio", 100.0),
                die_cost=_num(qs, "die_cost", 2.5),
                wafers_per_month=_num(qs, "wafers_per_month", 20000.0),
                scrap_ceiling=_num(qs, "scrap_ceiling", 0.25)))
        if head == "workorder" and len(parts) >= 2:
            return self._json(data.work_order(parts[1]))
        if head == "recon":
            return self._json(data.recon())
        if head == "rootcauses":
            return self._json(data.root_cause_table())
        if head == "focus":
            return self._json(data.focus_die())
        if head == "health":
            s = data.store()
            return self._json({"ok": True, "dataset": s.dataset,
                               "preds": len(s.preds), "model": s.model_name})
        return self._error(404, f"no API route '/api/{name}'")

    def _static(self, route: str):
        if route in ("/", ""):
            route = "/index.html"

        if route.startswith("/figures/"):
            target = (FIGURES / route[len("/figures/"):]).resolve()
            base = FIGURES.resolve()
        else:
            target = (STATIC / route.lstrip("/")).resolve()
            base = STATIC.resolve()

        if base not in target.parents and target != base:
            return self._error(403, "outside the served root")
        if not target.is_file():
            return self._error(404, f"no such file '{route}'")

        ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype == "application/javascript":
            ctype += "; charset=utf-8"
        cache = "no-store" if target.suffix in {".html", ".js", ".css"} else "max-age=600"
        self._send(200, target.read_bytes(), ctype, cache)


def main() -> None:
    ap = argparse.ArgumentParser(description="WaferLens OS mission-control server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--dataset", default="full", choices=["smoke", "full"])
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    data.DATASET = args.dataset
    print(f"WaferLens OS | loading '{args.dataset}' artifacts...")
    s = data.store(args.dataset)
    print(f"  model      {s.model_name}")
    print(f"  dies       {len(s.dies):,} ({len(s.preds):,} eligible) "
          f"across {s.dies['wafer_id'].nunique()} wafers")
    print(f"  threshold  {s.threshold:.4f}")

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    httpd.flags = ["--verbose"] if args.verbose else []
    print(f"\n  ready  ->  http://{args.host}:{args.port}\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
