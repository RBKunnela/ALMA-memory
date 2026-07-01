# Maia <-> ALMA Memory Adapter (Phase 1)

Backs the **Maia** gateway's memory with [ALMA](../../README.md). The
iMetaClaw / OpenGlasses app talks to the Maia gateway over WebSocket JSON-RPC;
this adapter implements the two memory methods on top of ALMA.

> **Scope: Phase 1 only — facts-first read + write.** Outcome/strategy/confidence
> enrichment (Phase 3) and RN parity / cross-device / HIPAA pruning (Phase 4) are
> deliberately **not** built here. See [Deferred](#deferred).

## Device contract (the shapes you must honor)

Verified in `OpenGlasses/Sources/Services/OpenClawBridge.swift:1172-1206`:

| Method         | Params                          | Success response                                                  |
| -------------- | ------------------------------- | ----------------------------------------------------------------- |
| `memory.query` | `{query: string, limit: int}`   | `{"ok": true, "payload": {"results": [{"content": string}, ...]}}` |
| `memory.store` | `{content: string, metadata: object}` | `{"ok": true}`                                              |

The adapter never raises into the RPC layer. On any failure it returns
`{"ok": false, "error": "..."}`, and **empty memory is a success with empty
results**, not an error.

## Files

| File                          | Purpose                                                        |
| ----------------------------- | -------------------------------------------------------------- |
| `alma_gateway_adapter.py`     | The two handlers + `build_alma()` factory. Framework-agnostic. |
| `spike_roundtrip.py`          | Runnable Phase-0 spike: store 2 facts, query them back.        |
| `test_alma_gateway_adapter.py`| Unit tests against a fake in-memory ALMA (no model/network).   |

## Interface

```python
from integrations.maia.alma_gateway_adapter import (
    build_alma, handle_memory_query, handle_memory_store,
)

alma = build_alma()  # ALMA.quickstart() local backend; or build_alma("/path/config.yaml")

handle_memory_query({"query": "deploy steps", "limit": 10}, alma=alma)
# -> {"ok": True, "payload": {"results": [{"content": "..."}, ...]}}

handle_memory_store({"content": "Caddy fronts :3600", "metadata": {"persona": "maia"}}, alma=alma)
# -> {"ok": True}
```

- `agent` defaults to `"maia"`. `metadata.persona` (when present) overrides the
  ALMA agent/scope per call. A `user_id` / `userId` / `user` in metadata is
  threaded into `retrieve()` when present.
- The adapter has **no top-level `alma` import** — only `build_alma()` imports it
  lazily — so the module imports cleanly without `alma-memory` installed, and the
  unit tests run against a fake.

### Import path

The module is `integrations.maia.alma_gateway_adapter`. Import it relative to the
repo root (the directory containing `integrations/`). On the VPS, either run the
gateway from the repo root or add the repo root to `PYTHONPATH`:

```bash
export PYTHONPATH=/opt/maia/alma-integration:$PYTHONPATH
```

## Phase-0 spike (do this first on the VPS)

```bash
python3 -m venv /opt/maia/venv && source /opt/maia/venv/bin/activate
pip install 'alma-memory[local]'        # SQLite + sentence-transformers, zero keys

# REPL round trip:
python - <<'PY'
from alma import ALMA
alma = ALMA.quickstart(project_id="maia")
alma.add_domain_knowledge(agent="maia", domain="maia_memory", fact="hello from maia")
print([dk.fact for dk in alma.retrieve(task="hello", agent="maia", top_k=5).domain_knowledge])
PY

# Or run the bundled spike (from the repo root):
python integrations/maia/spike_roundtrip.py
```

The first `quickstart()` downloads the sentence-transformers model (~80 MB);
subsequent runs are offline. SQLite lands under `./.alma/alma.db`.

## Wiring into the Maia gateway dispatcher

The Maia gateway lives on the user's VPS (not in any repo), listening on
`127.0.0.1:3600` behind Caddy on `:443`. Build one ALMA instance at startup and
route the two methods to the handlers. Sketch:

```python
from integrations.maia.alma_gateway_adapter import (
    build_alma, handle_memory_query, handle_memory_store,
)

ALMA_INSTANCE = build_alma()  # once, at process start

def dispatch(method: str, params: dict) -> dict:
    if method == "memory.query":
        return handle_memory_query(params, alma=ALMA_INSTANCE)
    if method == "memory.store":
        return handle_memory_store(params, alma=ALMA_INSTANCE)
    # ... existing methods ...
```

Notes:
- Build ALMA **once** and reuse it; `quickstart()` eagerly loads the embedder.
- ALMA is synchronous. If the dispatcher is `async`, run the handlers in a
  thread executor (`loop.run_in_executor(...)`) so embedding work does not block
  the WebSocket event loop.
- Keep `.alma/` on persistent disk so memory survives restarts; back it up.

## Tests

From the repo root (use the project venv if present, e.g. `/tmp/alma-venv`):

```bash
python -m pytest integrations/maia/ -q
```

Tests use a **fake in-memory ALMA**, so they pass without `sentence-transformers`
or network access. They cover query shape, empty-memory path, persona routing,
store ok/persist, bad-params -> `{ok: False}`, backend-failure isolation, and a
store->query round trip.

## Deferred

These are explicit TODOs, intentionally out of Phase 1:

- **Phase 3 — schema/contract enrichment.** Thread task outcome, `strategy_used`,
  and confidence through the RPC and call `alma.learn(...)` (and surface
  heuristic confidence in `memory.query` results). Today `memory.store` only
  writes facts via `add_domain_knowledge`; `learn()` is not invoked.
- **Phase 4 — RN parity / cross-device.** Match the React Native client's memory
  shape and reconcile per-device vs shared memory scopes.
- **Phase 4 — HIPAA server-side pruning.** Server-side retention/redaction of PHI
  before persistence; ALMA scope/TTL policy enforcement on stored content.
```
