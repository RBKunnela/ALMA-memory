# Atlas gaps closure — Chefe 561 / Code-Hub 1624

**Date:** 2026-08-05  
**Input:** Agent Memory Atlas report (neoneye / Claude Opus code read of commit `164d2e3e`) + Chefe email thank-you + instruction that Code-Hub owns implement after planning.

## Closed

| Gap | Fix |
|-----|-----|
| G1 Verification not persisted | Columns + `update_memory_verification` + VerifiedRetriever.storage |
| G2 Anti-patterns not write-path | `write_guard` on `LearningProtocol.learn` |
| G3 Dual schema | Migration v1.2.0 sqlite + postgresql |
| G4 Forget without audit | `alma_forget_audit` + ForgettingEngine hook |
| G5 LICENSE | A0 commit `7ce60ab` |

## Config

- `ALMA_ANTI_PATTERN_WRITE_GUARD=1` (default) — set `0` to disable.

## VerifiedRetriever usage (persist)

```python
retriever = VerifiedRetriever(
    retrieval_engine=engine,
    storage=storage,  # required for G1 persist
    config=VerificationConfig(enabled=True),
)
results = retriever.retrieve_verified(query=..., agent=..., project_id=...)
# list contradicted later:
storage.list_by_verification_status(project_id, "contradicted", "outcomes")
```

## Tests

```bash
pytest tests/unit/test_atlas_gaps_561.py -q
```
