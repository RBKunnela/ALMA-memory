# ALMA + paid stacks (ecosystem — not included in OSS)

**ALMA** is **MIT open source** (`pip install alma-memory`).  
It does **not** include proprietary products. That is intentional: paid material must not ship for free inside ALMA.

---

## What ALMA is (complete for **memory**)

ALMA alone is a full **learning memory** layer for any agent:

- Retrieve strategies, outcomes, anti-patterns before a task  
- Learn after success/failure  
- Your DB (SQLite / Postgres / Qdrant / …)  
- Cross-platform (Claude, OpenClaw, custom agents)  

**You do not need LAG, AWP, or PayBotFin to use ALMA.**  
If you only want agents that **remember and improve**, ALMA is enough.

---

## Two different jobs

| Job | Product | Required for memory? |
|-----|---------|----------------------|
| **Memory / learning** — what worked, what to avoid next time | **ALMA** (OSS) | **Yes — this is ALMA** |
| **Accountability** — prove what agents did, map the system, gate value | **LAG + AWP + PayBotFin** (paid / commercial) | **No** — optional security & ops layers |

Mixing these in one OSS package would either break ALMA’s MIT model or give away paid IP. So they stay **separate products** that can work **together**.

---

## What is **LAG**? (paid)

**LAG** = *Living Architecture Graph* — a **paid / private** product.

It answers: *what exists in my agentic system right now?*

- Agents, skills, tools, workflows as **entities**  
- **Edges** (who uses what, who produces what)  
- Health / orphans / topology over time  

**Why it matters next to ALMA:**  
ALMA stores *lessons* (“blue-green worked”). LAG stores *structure* (“this agent called that tool in that engagement”). Together you get **smarter agents** (ALMA) plus **an auditable map of the fleet** (LAG).  

**Not required** for retrieve/learn. Only for **architecture accountability** and ops visibility.

---

## What is **PayBotFin**? (paid)

**PayBotFin** is a **paid / private** product for **value and authorization receipts**.

It answers: *was this act authorized, metered, or commercially settled — with a numbered proof?*

- Authorize / allow paths  
- Numbered commercial receipts  
- Ties to witness/ledger patterns for high-stakes agent actions  

**Why it matters next to ALMA:**  
ALMA records *whether a strategy worked*. PayBotFin records *whether an act was allowed and receipted* (money, policy, customer proof). Together: **learning** + **commercial accountability**.  

**Not required** for memory. Only when you need **value/authorization accountability**.

---

## What is **AWP**? (protocol + commercial paths)

**AWP** (*Agent Witness Protocol*) is about **tamper-evident receipts of acts** — offline-verifiable witness records.  

Open verification pieces can exist for ecosystem trust; **commercial multi-tenant issuance** and full production witness paths are **not** bundled into ALMA.  

**Why next to ALMA:** memory says “we learned X”; AWP says “act Y happened and can be verified.”  

**Not required** for memory — **accountability** only.

---

## How they fit (optional paid spine)

```
  Agent work
      │
      ├─ memory:   ALMA.retrieve / ALMA.learn     ← enough for “smarter next time”
      │
      └─ optional accountability (paid stacks):
            LAG     → map who/what in the system
            AWP     → witness that the act happened
            PayBotFin → authorize / value receipt
```

| If you need… | Use |
|--------------|-----|
| Persistent learning memory only | **ALMA** |
| + Fleet / architecture truth | **+ LAG** (paid) |
| + Cryptographic act witness | **+ AWP** (commercial path) |
| + Value / authorize receipts | **+ PayBotFin** (paid) |

**ALMA is complete for memory options.**  
**LAG, AWP, and PayBotFin are extra security and accountability layers** — not memory features.

---

## Benefits of using them **together** (still separate products)

1. **Smarter agents** (ALMA) without opening the ledger IP.  
2. **Accountability** for multi-agent production (LAG map + AWP witness + PayBotFin value).  
3. Private control planes (e.g. ParviClaw Core) can **call** ALMA as an OSS client — ALMA never depends on paid packages.  
4. Clear commercial story: free memory layer · paid accountability spine.

---

## What we will never do in this repo

- Depend on or vendor **LAG** or **PayBotFin**  
- Require paid APIs for `pip install alma-memory`  
- Market ALMA as “includes LAG/PayBotFin”  
- Put paid material inside MIT by “convenience”

Commercial inquiries for paid stacks: **https://agentictestari.com**

---

*Chefe 1932 · 1935 · OSS boundary 1927/1928*
