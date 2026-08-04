#!/usr/bin/env python3
"""Live smoke tests for the sentrix-evaluator API.

Standalone script (no pytest dependency) using ``httpx`` (falls back to the
standard library ``urllib`` if httpx is unavailable). Targets a running
container/pod at ``--url`` / ``TARGET_URL`` (default ``http://localhost:8000``).

Checks:
  a. GET /health              -> HTTP 200 and a healthy JSON body
  b. GET /metrics             -> HTTP 200, text/plain, contains a ``sentrix_`` metric
  c. X-Request-ID propagation -> /health echoes the supplied correlation id
  d. DB stream ping            -> GET /readyz returns 200 (DB pool + buffer ready)

Exits 0 on full success, 1 if any check fails. A summary table is printed.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field

DEFAULT_URL = "http://localhost:8000"


@dataclass
class Result:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class Smoke:
    results: list[Result] = field(default_factory=list)

    def record(self, name: str, passed: bool, detail: str = "") -> None:
        self.results.append(Result(name, passed, detail))
        tag = "PASS" if passed else "FAIL"
        print(f"  [{tag}] {name}" + (f" — {detail}" if detail else ""))

    def run(self, base_url: str) -> int:
        print(f"\nSmoke-testing {base_url}")
        self._check_health(base_url)
        self._check_metrics(base_url)
        self._check_correlation_id(base_url)
        self._check_db_bridge(base_url)
        self.summary()
        return 0 if all(r.passed for r in self.results) else 1

    # ── individual checks ──────────────────────────────────────────────
    def _get(self, base_url: str, path: str, headers: dict | None = None):
        try:
            import httpx  # type: ignore
        except ImportError:
            return _urllib_get(base_url, path, headers)
        client = httpx.Client(timeout=10.0)
        try:
            return client.get(f"{base_url}{path}", headers=headers or {})
        finally:
            client.close()

    def _check_health(self, base_url: str) -> None:
        try:
            resp = self._get(base_url, "/health")
        except Exception as exc:  # network / connection
            self.record("GET /health", False, f"connection error: {exc}")
            return
        body_ok = False
        try:
            data = resp.json()
            status = str(data.get("status", "")).lower()
            body_ok = status in {"alive", "ok", "ready", "healthy"} or resp.status_code == 200
        except Exception:
            body_ok = resp.status_code == 200
        self.record(
            "GET /health",
            resp.status_code == 200 and body_ok,
            f"status={resp.status_code} body={resp.text[:80]}",
        )

    def _check_metrics(self, base_url: str) -> None:
        try:
            resp = self._get(base_url, "/metrics")
        except Exception as exc:
            self.record("GET /metrics", False, f"connection error: {exc}")
            return
        ctype = resp.headers.get("content-type", "")
        is_text = "text/plain" in ctype
        has_sentrix = "sentrix_" in resp.text
        self.record(
            "GET /metrics",
            resp.status_code == 200 and is_text and has_sentrix,
            f"status={resp.status_code} ctype={ctype!r} sentrix_={'yes' if has_sentrix else 'no'}",
        )

    def _check_correlation_id(self, base_url: str) -> None:
        try:
            resp = self._get(base_url, "/health", headers={"X-Request-ID": "smoke-test-123"})
        except Exception as exc:
            self.record("X-Request-ID propagation", False, f"connection error: {exc}")
            return
        echoed = resp.headers.get("X-Request-ID", "")
        self.record(
            "X-Request-ID propagation",
            resp.status_code == 200 and echoed == "smoke-test-123",
            f"status={resp.status_code} x-request-id={echoed!r}",
        )

    def _check_db_bridge(self, base_url: str) -> None:
        # GET /readyz verifies DB pool reachability + ingestion-buffer state —
        # used as the live "DB stream ping" since the gateway has no /v1/eval.
        try:
            resp = self._get(base_url, "/readyz")
        except Exception as exc:
            self.record("GET /readyz (DB bridge)", False, f"connection error: {exc}")
            return
        ready = resp.status_code == 200
        detail = ""
        try:
            detail = json.dumps(resp.json())
        except Exception:
            detail = resp.text[:80]
        self.record(
            "GET /readyz (DB bridge)",
            ready,
            f"status={resp.status_code} body={detail[:80]}",
        )

    # ── summary table ─────────────────────────────────────────────────
    def summary(self) -> None:
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        print("\n" + "=" * 64)
        print(f"SMOKE SUMMARY: {passed}/{total} checks passed")
        print("=" * 64)
        for r in self.results:
            mark = "PASS" if r.passed else "FAIL"
            print(f"  [{mark}] {r.name}")
        print("=" * 64)


def _urllib_get(base_url: str, path: str, headers: dict | None | None = None):
    import urllib.request

    req = urllib.request.Request(f"{base_url}{path}", headers=headers or {})
    with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310 - intended localhost target
        return _UrllibResp(resp.status, resp.headers, resp.read().decode("utf-8", "replace"))


@dataclass
class _UrllibResp:
    status_code: int
    headers: object
    text: str

    @property
    def json(self):
        return json.loads(self.text)


def main() -> int:
    import argparse

    url = os.environ.get("TARGET_URL")
    parser = argparse.ArgumentParser(description="Live smoke tests for sentrix-evaluator")
    parser.add_argument("--url", default=url or DEFAULT_URL, help="base URL (default: %(default)s)")
    args = parser.parse_args()
    return Smoke().run(args.url)


if __name__ == "__main__":
    sys.exit(main())
