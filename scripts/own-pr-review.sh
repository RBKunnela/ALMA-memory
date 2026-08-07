#!/usr/bin/env bash
# Own PR reviewer — Chefe GO A (1793) · no CodeRabbit/Codex SaaS
# Uses LLM-Hub (local free models) + gh. Run on fleet host (LLM-Hub reachable).
# Usage:
#   OWN_REVIEW_REPO=RBKunnela/ALMA-memory OWN_REVIEW_PR=44 bash scripts/own-pr-review.sh
#   bash scripts/own-pr-review.sh 44
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

REPO="${OWN_REVIEW_REPO:-RBKunnela/ALMA-memory}"
PR="${1:-${OWN_REVIEW_PR:-}}"
LLM_BASE="${LLM_HUB_URL:-http://127.0.0.1:4000}"
MODEL="${OWN_REVIEW_MODEL:-qwen-7b-local}"
MAX_DIFF_CHARS="${OWN_REVIEW_MAX_DIFF:-48000}"

if [[ -z "$PR" ]]; then
  echo "Usage: $0 <pr-number>  or set OWN_REVIEW_PR"
  exit 2
fi

command -v gh >/dev/null || { echo "gh required"; exit 1; }
command -v curl >/dev/null || { echo "curl required"; exit 1; }
command -v python3 >/dev/null || { echo "python3 required"; exit 1; }

echo "[own-review] repo=$REPO pr=$PR model=$MODEL hub=$LLM_BASE"

TITLE=$(gh pr view "$PR" --repo "$REPO" --json title -q .title)

TMPD=$(mktemp -d)
trap 'rm -rf "$TMPD"' EXIT

gh pr diff "$PR" --repo "$REPO" >"$TMPD/diff.txt" 2>/dev/null || true
if [[ ! -s "$TMPD/diff.txt" ]]; then
  echo "[own-review] empty diff — skip"
  exit 0
fi

python3 - <<PY
from pathlib import Path
p = Path("$TMPD/diff.txt")
t = p.read_text(errors="replace")
max_c = int("$MAX_DIFF_CHARS")
if len(t) > max_c:
    t = t[:max_c] + "\n\n… [diff truncated for LLM context] …\n"
Path("$TMPD/diff_trim.txt").write_text(t)
print(f"[own-review] diff_chars={len(t)}")
PY

# Escape title for embedding in Python string (via env + file avoids shell hell)
export OWN_REVIEW_TMPD="$TMPD" OWN_REVIEW_TITLE="$TITLE" OWN_REVIEW_PR="$PR" \
  OWN_REVIEW_REPO_NAME="$REPO" OWN_REVIEW_MODEL_NAME="$MODEL"

python3 - <<'PY'
import json
import os
from pathlib import Path

tmp = Path(os.environ["OWN_REVIEW_TMPD"])
diff = (tmp / "diff_trim.txt").read_text(errors="replace")
title = os.environ["OWN_REVIEW_TITLE"]
pr = os.environ["OWN_REVIEW_PR"]
repo = os.environ["OWN_REVIEW_REPO_NAME"]
model = os.environ["OWN_REVIEW_MODEL_NAME"]
payload = {
    "model": model,
    "messages": [
        {
            "role": "system",
            "content": (
                "You are the RBKunnela own PR reviewer (Agentic Testari). "
                "CodeRabbit/Codex SaaS are OFF (Chefe 1792). "
                "Review the git diff. Markdown only with sections: "
                "## Verdict (PASS|PASS_WITH_NOTES|REQUEST_CHANGES), ## Summary, "
                "## Blocking, ## Suggestions, ## Security. "
                "Never claim free of vulnerabilities."
            ),
        },
        {
            "role": "user",
            "content": f"Repo: {repo}\nPR #{pr}: {title}\n\nDiff:\n```\n{diff}\n```",
        },
    ],
    "temperature": 0.2,
    "max_tokens": 1800,
}
(tmp / "payload.json").write_text(json.dumps(payload))
PY

HTTP=$(curl -sS -o "$TMPD/resp.json" -w "%{http_code}" \
  -X POST "$LLM_BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d @"$TMPD/payload.json" || echo "000")

if [[ "$HTTP" != "200" ]]; then
  echo "[own-review] LLM-Hub HTTP $HTTP — posting fallback checklist"
  BODY=$(cat <<MD
## Own PR review (fallback — LLM-Hub unreachable)

**Issuer:** Agentic Testari · automated gate  
**PR:** #$PR · \`$REPO\`  
**Model:** $MODEL · hub \`$LLM_BASE\` → HTTP $HTTP

### Manual / mechanical checks
- [ ] CI green (lint, tests, dogfood)
- [ ] No secrets in diff
- [ ] Sentinel dogfood smoke when applicable
- [ ] George gatekeeper path for merge (DevOps)

_CodeRabbit/Codex SaaS disabled (Chefe 1792). Re-run when LLM-Hub is up._
MD
)
else
  BODY=$(python3 - <<PY
import json
from pathlib import Path
r = json.loads(Path("$TMPD/resp.json").read_text())
try:
    content = r["choices"][0]["message"]["content"]
except Exception:
    content = (
        "_(could not parse LLM response)_\n\n\`\`\`\n"
        + Path("$TMPD/resp.json").read_text()[:2000]
        + "\n\`\`\`"
    )
header = """## Own PR review (Agentic Testari / LLM-Hub)

**Repo:** $REPO · **PR:** #$PR  
**Model:** \`$MODEL\` · **CodeRabbit/Codex:** OFF (Chefe 1792)

---
"""
print(header + content)
PY
)
fi

MARKER="<!-- rbk-own-pr-review -->"
EXISTING=$(gh api "repos/$REPO/issues/$PR/comments" --jq '.[].body' 2>/dev/null | grep -c "$MARKER" || true)
FULL="${MARKER}
${BODY}

---
_Continuous dogfood + own review · https://agentictestari.com_"

if [[ "${EXISTING:-0}" -gt 0 ]]; then
  echo "[own-review] comment already present — skip duplicate (set OWN_REVIEW_FORCE=1 to repost)"
  if [[ "${OWN_REVIEW_FORCE:-0}" != "1" ]]; then
    exit 0
  fi
fi

# Body via file — avoid gh CLI arg length / quote issues
printf '%s\n' "$FULL" >"$TMPD/comment.md"
gh pr comment "$PR" --repo "$REPO" --body-file "$TMPD/comment.md"
echo "[own-review] posted comment on $REPO#$PR"
mkdir -p .sentinel-dogfood
echo "{\"pr\":$PR,\"repo\":\"$REPO\",\"model\":\"$MODEL\",\"status\":\"posted\",\"http\":$HTTP}" \
  >.sentinel-dogfood/last-own-review.json
