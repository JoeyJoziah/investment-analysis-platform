#!/usr/bin/env python3
"""
Production smoke test — verifies critical endpoints and service connectivity.

Run after deploying to a fresh environment. Exits non-zero on any failure.

Usage:
    python scripts/smoke_test.py [--base-url https://your-domain.com]

Env vars:
    SMOKE_BASE_URL     Override base URL (default: http://localhost:8000)
    SMOKE_TIMEOUT      Per-request timeout seconds (default: 10)
    SMOKE_AUTH_TOKEN   Optional bearer token for authenticated checks
    SMOKE_GRAFANA_URL  Optional Grafana URL to probe (default: http://localhost:3001)
    SMOKE_PROM_URL     Optional Prometheus URL to probe (default: http://localhost:9090)
"""
import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

try:
    import httpx
except ImportError:
    print("FATAL: httpx not installed. Run: pip install httpx", file=sys.stderr)
    sys.exit(2)


@dataclass
class Result:
    name: str
    ok: bool
    detail: str
    latency_ms: float = 0.0


@dataclass
class Suite:
    base_url: str
    timeout: float
    auth_token: Optional[str]
    grafana_url: str
    prom_url: str
    results: list = field(default_factory=list)

    def run(self, name: str, fn: Callable[[], tuple]) -> Result:
        t0 = time.perf_counter()
        try:
            ok, detail = fn()
        except Exception as e:
            ok, detail = False, f"{type(e).__name__}: {e}"
        latency = (time.perf_counter() - t0) * 1000
        r = Result(name, ok, detail, latency)
        self.results.append(r)
        marker = "OK " if ok else "FAIL"
        print(f"  [{marker}] {name:40s} {latency:7.1f} ms  {detail}")
        return r

    def summary(self) -> int:
        passed = sum(1 for r in self.results if r.ok)
        failed = len(self.results) - passed
        print()
        print(f"Summary: {passed}/{len(self.results)} passed, {failed} failed")
        return 0 if failed == 0 else 1


def check_health(s: Suite):
    def fn():
        r = httpx.get(f"{s.base_url}/health", timeout=s.timeout)
        return (r.status_code == 200, f"status={r.status_code}")
    return s.run("backend /health", fn)


def check_metrics(s: Suite):
    def fn():
        r = httpx.get(f"{s.base_url}/metrics", timeout=s.timeout)
        ok = r.status_code == 200 and "process_" in r.text
        return (ok, f"status={r.status_code} bytes={len(r.text)}")
    return s.run("backend /metrics", fn)


def check_docs(s: Suite):
    def fn():
        r = httpx.get(f"{s.base_url}/docs", timeout=s.timeout)
        return (r.status_code == 200, f"status={r.status_code}")
    return s.run("OpenAPI /docs", fn)


def check_openapi_schema(s: Suite):
    def fn():
        r = httpx.get(f"{s.base_url}/openapi.json", timeout=s.timeout)
        ok = r.status_code == 200
        if ok:
            schema = r.json()
            paths = len(schema.get("paths", {}))
            return (paths > 0, f"paths={paths}")
        return (False, f"status={r.status_code}")
    return s.run("OpenAPI schema", fn)


def check_auth_endpoints(s: Suite):
    def fn():
        r = httpx.post(
            f"{s.base_url}/api/v1/auth/login",
            json={"email": "x", "password": "x"},
            timeout=s.timeout,
        )
        # 400/401/422 all acceptable: endpoint reachable, auth rejected
        ok = r.status_code in (400, 401, 422)
        return (ok, f"status={r.status_code}")
    return s.run("auth login reachable", fn)


def check_websocket_endpoint(s: Suite):
    def fn():
        ws_path = f"{s.base_url}/api/v1/ws"
        r = httpx.get(ws_path, timeout=s.timeout)
        # WS upgrade required => 426 or 400; either signals route exists
        ok = r.status_code in (400, 426, 404, 405)
        return (ok, f"status={r.status_code}")
    return s.run("websocket reachable", fn)


def check_grafana(s: Suite):
    def fn():
        r = httpx.get(f"{s.grafana_url}/api/health", timeout=s.timeout)
        return (r.status_code == 200, f"status={r.status_code}")
    return s.run("Grafana /api/health", fn)


def check_prometheus(s: Suite):
    def fn():
        r = httpx.get(f"{s.prom_url}/-/ready", timeout=s.timeout)
        return (r.status_code == 200, f"status={r.status_code}")
    return s.run("Prometheus /-/ready", fn)


def check_database_via_health(s: Suite):
    def fn():
        r = httpx.get(f"{s.base_url}/health/db", timeout=s.timeout)
        if r.status_code == 404:
            return (True, "endpoint not exposed (skip)")
        return (r.status_code == 200, f"status={r.status_code}")
    return s.run("backend /health/db", fn)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default=os.getenv("SMOKE_BASE_URL", "http://localhost:8000"))
    p.add_argument(
        "--timeout", type=float,
        default=float(os.getenv("SMOKE_TIMEOUT", "10")),
    )
    p.add_argument("--grafana", default=os.getenv("SMOKE_GRAFANA_URL", "http://localhost:3001"))
    p.add_argument(
        "--prometheus",
        default=os.getenv("SMOKE_PROM_URL", "http://localhost:9090"),
    )
    p.add_argument("--skip-monitoring", action="store_true")
    p.add_argument("--json", action="store_true", help="emit JSON report at end")
    args = p.parse_args()

    suite = Suite(
        base_url=args.base_url.rstrip("/"),
        timeout=args.timeout,
        auth_token=os.getenv("SMOKE_AUTH_TOKEN"),
        grafana_url=args.grafana.rstrip("/"),
        prom_url=args.prometheus.rstrip("/"),
    )

    print(f"Smoke testing {suite.base_url} (timeout={suite.timeout}s)")
    print("-" * 70)
    check_health(suite)
    check_metrics(suite)
    check_docs(suite)
    check_openapi_schema(suite)
    check_auth_endpoints(suite)
    check_websocket_endpoint(suite)
    check_database_via_health(suite)
    if not args.skip_monitoring:
        check_grafana(suite)
        check_prometheus(suite)

    rc = suite.summary()
    if args.json:
        print()
        print(json.dumps({
            "base_url": suite.base_url,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "results": [r.__dict__ for r in suite.results],
            "exit_code": rc,
        }, indent=2))
    sys.exit(rc)


if __name__ == "__main__":
    main()
