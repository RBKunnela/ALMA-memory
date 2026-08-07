#!/usr/bin/env bash
# Sentinel Hub dogfood smoke — ALMA (Chefe 1775)
# Runs on every PR/push and weekly cron. Non-destructive. Exit 0 = pass.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
FAIL=0
note() { echo "[dogfood] $*"; }
fail() { echo "[dogfood] FAIL: $*"; FAIL=1; }

note "ALMA Sentinel dogfood smoke — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
note "Issuer: Agentic Testari Sentinel Hub · https://agentictestari.com"

# 1) Security unit tests (MCP bind, fail-closed, headers file, write-guard doors)
if command -v python3 >/dev/null; then
  note "pytest security suites"
  python3 -m pytest -q \
    tests/unit/test_security_close_1770.py \
    tests/unit/test_atlas_gaps_561.py \
    --tb=line || fail "security pytest"
else
  fail "python3 missing"
fi

# 2) Static invariants (code SoT)
note "static invariants"
python3 - <<'PY' || exit 1
from pathlib import Path
root = Path(".")
errors = []
main = (root / "alma/mcp/__main__.py").read_text()
server = (root / "alma/mcp/server.py").read_text()
if 'default="127.0.0.1"' not in main and "default='127.0.0.1'" not in main:
    errors.append("MCP HTTP default host must be 127.0.0.1")
if "ALMA_MCP_TOKEN" not in server:
    errors.append("MCP HTTP must document/require ALMA_MCP_TOKEN for bind-all")
if "write_guard_fail_closed" not in (root / "alma/learning/write_guard.py").read_text():
    errors.append("write_guard_fail_closed missing")
if not (root / "alma/storage/write_guard_hooks.py").is_file():
    errors.append("storage write_guard_hooks.py missing (doors)")
if not (root / "site-docs/_headers").is_file():
    errors.append("site-docs/_headers missing")
hdr = (root / "site-docs/_headers").read_text()
for need in ("Strict-Transport-Security", "Content-Security-Policy", "X-Frame-Options"):
    if need not in hdr:
        errors.append(f"_headers missing {need}")
if errors:
    print("INVARIANT FAIL:")
    for e in errors:
        print(" -", e)
    raise SystemExit(1)
print("invariants OK")
PY

# 3) Optional live docs header check (non-fatal if offline)
if [[ "${DOGFOOD_LIVE_HEADERS:-0}" == "1" ]]; then
  note "live headers check alma-memory.pages.dev"
  python3 - <<'PY' || fail "live headers"
import urllib.request
req = urllib.request.Request(
    "https://alma-memory.pages.dev/",
    headers={"User-Agent": "AgenticTestari-Sentinel-Dogfood/1.0"},
)
with urllib.request.urlopen(req, timeout=20) as r:
    h = {k.lower(): v for k, v in r.headers.items()}
missing = [n for n in (
    "strict-transport-security",
    "content-security-policy",
    "x-frame-options",
) if n not in h]
if missing:
    print("LIVE HEADERS MISSING (post-deploy):", missing)
    raise SystemExit(1)
print("live headers OK")
PY
else
  note "skip live headers (set DOGFOOD_LIVE_HEADERS=1 after Pages deploy)"
fi

# 4) Emit stamp for artifacts / ads attribution
mkdir -p .sentinel-dogfood
cat > .sentinel-dogfood/last-run.json <<JSON
{
  "issuer": "Agentic Testari",
  "website": "https://agentictestari.com",
  "product": "Sentinel Hub",
  "repo": "RBKunnela/ALMA-memory",
  "profile": "internal-dogfood-smoke",
  "status": "$([ "$FAIL" -eq 0 ] && echo pass || echo fail)"
}
JSON

if [[ "$FAIL" -ne 0 ]]; then
  note "DOGFOOD FAIL"
  exit 1
fi
note "DOGFOOD PASS — security gates green"
note "Attribution: https://agentictestari.com"
exit 0
