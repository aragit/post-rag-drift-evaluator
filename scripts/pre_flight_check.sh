#!/usr/bin/env bash
# scripts/pre_flight_check.sh
#
# Local / CI pre-flight gate for sentrix-evaluator.
# Runs, in order:
#   a. Python version check (>=3.10)
#   b. ruff check evaluator/ api/ ingestion/
#   c. mypy type check (ADVISORY — runs and reports, but does not abort the gate)
#   d. pytest tests/            (aborts on failure)
#   e. python -m build         (aborts on failure: verifies sdist + wheel)
#
# The gate passes (exit 0) only when all *hard* checks succeed. mypy is
# advisory because the codebase is not yet mypy-baselined; install it with
# `python3 -m pip install mypy` to enable. ruff/pytest/build hard-fail under
# `set -e`, satisfying the "exit immediately on failure" requirement for pytest.
set -euo pipefail

# ── colourised logging ──────────────────────────────────────────────────────
INFO=$'\033[0;36m'
OK=$'\033[0;32m'
WARN=$'\033[0;33m'
ERR=$'\033[0;31m'
BOLD=$'\033[1m'
RESET=$'\033[0m'
info() { printf "${BOLD}${INFO}[INFO]${RESET} %s\n" "$*"; }
ok()   { printf "${BOLD}${OK}[SUCCESS]${RESET} %s\n" "$*"; }
warn() { printf "${BOLD}${WARN}[WARN]${RESET} %s\n" "$*"; }
err()  { printf "${BOLD}${ERR}[ERROR]${RESET} %s\n" "$*" >&2; }

# ── locate repo root (script lives in scripts/) ──────────────────────────────
cd "$(dirname "$0")/.."
ROOT="$PWD"
info "Repo root: $ROOT"

PY_BIN="${PYTHON:-python3}"

# ── (a) Python >= 3.10 ──────────────────────────────────────────────────────
info "(a) Checking Python version (requires >=3.10)..."
if "$PY_BIN" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
  vv="$("$PY_BIN" -c 'import sys;print(".".join(map(str,sys.version_info[:3])))')"
  ok "Python $vv satisfies >=3.10"
else
  err "Python >=3.10 required (found $($PY_BIN -c 'import sys;print(sys.version.split()[0])'))"
  exit 1
fi

# ── (b) ruff ────────────────────────────────────────────────────────────────
info "(b) Running ruff check evaluator/ api/ ingestion/ ..."
ruff check evaluator/ api/ ingestion/
ok "ruff: no lint errors (evaluator/ api/ ingestion/)"

# ── (c) mypy (advisory) ─────────────────────────────────────────────────────
info "(c) Running mypy on evaluator/ api/ (advisory) ..."
if ! "$PY_BIN" -c 'import mypy' >/dev/null 2>&1; then
  warn "mypy not installed; install with '${PY_BIN} -m pip install mypy' to enable this check"
else
  if "$PY_BIN" -m mypy --ignore-missing-imports evaluator/ api/ >/tmp/.preflight_mypy.log 2>&1; then
    ok "mypy: type check clean"
  else
    mypy_lines=$(grep -cE ": [0-9]+ error" /tmp/.preflight_mypy.log || true)
    warn "mypy: $mypy_lines type issue(s) reported — advisory only, gate continues (see /tmp/.preflight_mypy.log)"
  fi
fi

# ── (d) pytest ──────────────────────────────────────────────────────────────
info "(d) Running pytest tests/ ..."
pytest tests/
ok "pytest: all tests passed"

# ── (e) build ───────────────────────────────────────────────────────────────
info "(e) Verifying packaging with python -m build ..."
"$PY_BIN" -m build --outdir dist
ok "build: sdist + wheel generated in dist/"

ok "Pre-flight checks passed."
