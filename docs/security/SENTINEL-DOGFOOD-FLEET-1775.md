# Sentinel Hub dogfood fleet — cron + every update (Chefe 1775)

**Issuer:** Agentic Testari · https://agentictestari.com  
**Product:** Sentinel Hub  
**Ask:** Cron (weekly/monthly) **and/or every time we change a product** — scan with Sentinel Hub; wire into repos; eventually **all projects**.

## Cadence (recommended)

| Trigger | When | Depth |
|---------|------|--------|
| **On every PR / push to main** | Code changes | Dogfood **smoke** (unit security + invariants) — blocks merge if red |
| **Weekly cron** | e.g. Monday 06:00 UTC | Smoke + **live** surface checks (headers, public URLs) |
| **Monthly / major release** | Tag / release | Full L3 dogfood pack (findings.json + severity gate + report) |
| **Manual** | `workflow_dispatch` | Same as weekly or L3 |

“Every day” is optional later if noise is low; start **PR + weekly**.

## ALMA (pilot — this repo)

| Piece | Path |
|-------|------|
| Smoke script | `scripts/sentinel-dogfood-smoke.sh` |
| CI workflow | `.github/workflows/sentinel-dogfood.yml` |
| Attribution | stamp `.sentinel-dogfood/last-run.json` → agentictestari.com |

## All repos (rollout)

1. **Template workflow** in a shared action or copy `sentinel-dogfood.yml` per repo  
2. **Registry** of public surfaces (URL + repo) under lab: `sentinel-agentic-lab/docs/dogfood/fleet-registry.yaml`  
3. **Central runner** (lab or Hetzner) for full L3 packs; CI stays light  
4. **Maia STATUS** weekly rollup link for Chefe  

### Candidate fleet (first wave)

| Repo / product | Surface |
|----------------|---------|
| ALMA-memory | PyPI + pages.dev + MCP |
| paybotfin-* | APIs / dashboards |
| parvisight | engine + demos |
| parviclaw-core | core APIs |
| lag | service |
| friendlyai-web / agentictestari | public sites |
| sentinel-agentic-lab | meta (scan the scanner — later) |

## Claims for ads (honest)

- “Every update to ALMA is security-gated with Agentic Testari Sentinel Hub”  
- Not: “free of all vulnerabilities forever”

## Implementation status

| Item | Status |
|------|--------|
| ALMA smoke + PR/weekly CI | **This PR** |
| Full L3 auto pack in CI | Backlog (heavy; keep monthly/manual) |
| Fleet registry + other repos | Next wave after ALMA green on main |

— Pedro · Chefe 1775  
