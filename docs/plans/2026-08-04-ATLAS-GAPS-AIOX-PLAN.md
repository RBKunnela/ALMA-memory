# Plano AIOX — fechar gaps ALMA (atlas neoneye) · Chefe 561

**Authority:** Chefe voice **561**  
**Owner plano:** Orion (AIOX Master) — analyst / architect / PM / PO / SM  
**Implementação:** BMAD-Orch (method/stories) + Hermes-Pio (runtime/CI/deploy) + AIOX-dev se preciso  
**Pedro:** **não** nesta fila (HOLD cyber + busy Supabase/Railway)  
**Repo:** https://github.com/RBKunnela/ALMA-memory  
**Input:** Agent Memory Atlas · [ALMA report](https://neoneye.github.io/agent-memory-atlas/systems/alma-memory/) · commit analisado `164d2e3e…`

---

## 1) Analyst — problema

ALMA é forte em anti-patterns (`why_bad` + `better_alternative`) e scope SQL.  
O atlas apontou **near-misses epistêmicos e operacionais** que limitam trust/audit e a correção real.

### Gaps (prioridade)

| ID | Gap | Severidade | Evidência atlas / local |
|----|-----|------------|-------------------------|
| **G1** | `VerificationStatus` calculado no retrieve e **não persistido** | P0 | verification.py; atlas §1, §9 |
| **G2** | Anti-patterns = guidance, **não write-guard** | P0 | anti_patterns não consultados no write |
| **G3** | Schema **Postgres + SQLite** duplicado sem teste de paridade | P0 | cli.py DDL vs sqlite_local |
| **G4** | `ForgettingEngine` poda **sem audit** do removido | P1 | forgetting.py |
| **G5** | Arquivo **LICENSE** ausente no tree (só MIT no pyproject) | P1 | confirmado: sem LICENSE no clone |
| **G6** | Benchmarks claim vs suite documentada no CI | P2 | benchmarks/ presentes; validar CI |
| **G7** | (Opcional) Human review queue de CONTRADICTED | P2 | depende de G1 |

---

## 2) Architect — direção da solução

### G1 — Persist verification
- Colunas (ou tabela `alma_memory_verification`):  
  `verification_status`, `verification_method`, `verification_confidence`, `verification_reason`, `verified_at`, `contradicting_source`  
- Preencher no retrieve path **e** opcional background re-verify job  
- Queries: list CONTRADICTED for review; rank verified higher  

### G2 — Anti-pattern write guard
- Antes de `learn`/insert heuristic/outcome/domain: check similarity vs `alma_anti_patterns`  
- Se match forte: block ou force status `blocked_by_anti_pattern` + log  
- Config flag `ALMA_ANTI_PATTERN_WRITE_GUARD=1` default on em v1.x  

### G3 — Schema parity
- Single source of truth (migrations folder) gerando PG + SQLite  
- Teste CI: `test_schema_parity.py` compara colunas/indexes  

### G4 — Forget audit
- Tabela append-only `alma_forget_audit` (what, why, strategy, at, agent)  
- ForgettingEngine escreve antes de delete  

### G5 — LICENSE
- Adicionar `LICENSE` MIT no root alinhado ao pyproject  

### G6 — Benchmarks
- Documentar comando reproduce + job CI optional/nightly  

---

## 3) PM — epic e fases

**EPIC-ALMA-ATLAS-GAPS-001** — Close Agent Memory Atlas near-misses  

| Phase | Outcome | Owner code |
|-------|---------|------------|
| **A0** | LICENSE + CHANGELOG entry | Hermes/AIOX-dev quick |
| **A1** | Schema migrations G1+G3+G4 | Hermes + AIOX-dev |
| **A2** | Persist verification + API/MCP surface | Hermes + AIOX-dev |
| **A3** | Write-guard anti-patterns | Hermes + AIOX-dev |
| **A4** | Tests + schema parity CI | BMAD TEA + Hermes CI |
| **A5** | Docs + release note; Discord reply optional | Orion/Maia |

**DoD epic:** atlas near-misses G1–G5 fechados ou explicitamente deferred com ADR; CI green; tag/release patch.

---

## 4) PO — acceptance criteria (resumo)

### Story A0-LICENSE
- [x] `LICENSE` MIT no root  (`7ce60ab`)
- [x] README aponta para o arquivo  

### Story A1-SCHEMA
- [x] Migrations versionadas PG + SQLite  (`v1_2_0_atlas_gaps.py`)
- [x] Colunas verification + tabela forget_audit  
- [x] Upgrade path documentado (SQLite existing DBs via `_ensure_atlas_gap_schema`)  

### Story A2-PERSIST-VERIFY
- [x] Após retrieve com verification, status gravado  (`VerifiedRetriever` + storage)
- [x] MCP/API pode filtrar por status  (`list_by_verification_status`)
- [x] Teste: contradicted reaparece no DB  

### Story A3-WRITE-GUARD
- [x] Learn blocked/flagged when anti-pattern matches  
- [x] Config on/off  (`ALMA_ANTI_PATTERN_WRITE_GUARD`)
- [x] Teste unitário com anti-pattern planted  

### Story A4-PARITY-CI
- [x] `pytest` schema parity  (`test_schema_parity_sqlite_columns`)
- [ ] CI job green  (local suite green; full CI on push)

### Story A5-DOCS
- [x] ADR ou docs/architecture gap-closure  
- [x] CHANGELOG  

**Code-Hub implement:** 2026-08-05 Chefe 1624 (Pedro)

---

## 5) SM — ordem de sprint (sugerida)

**Sprint 1 (rápido, 1–3 dias):** A0 + A1 + A2 core  
**Sprint 2:** A3 + A4  
**Sprint 3:** A5 + polish + optional G7  

Gates: harness AIOX (design→test→code). BMAD formaliza stories se faltar detalhe.

---

## 6) QA — matriz mínima

| Case | Expect |
|------|--------|
| Retrieve contradicted twice | Same row status persisted |
| Learn known anti-pattern | Block or flag |
| Forget row | Audit row exists |
| Schema parity test | Pass |
| No secret in logs | Pass |

---

## 7) Command path (Chefe 561)

```
Orion (plan/gates)
  → BMAD-Orch: stories/AC pack from this plan
  → Hermes-Pio: implement + CI + release packaging
  → DevOps/Gage: push/tag when George green (if PR)
  → Maia: only if multi-agent Chefe consolidate
```

**Não** dual-spam Chefe com status de código intermediário — Orion reporta marcos.

---

## 8) Fora de escopo (este epic)

- Reescrever todo o ALMA  
- Competir com todo o atlas em 7/7 marks de uma vez  
- Hosted multi-tenant enterprise compliance  
- Misturar com cyber-eval Q-Factory  

---

## 9) Referências

- Internal atlas research note (not in this repo)  
- This repository  
- Atlas: https://neoneye.github.io/agent-memory-atlas/systems/alma-memory/  

— Orion · AIOX Master · Chefe 561  
