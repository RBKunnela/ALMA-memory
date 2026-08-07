# ALMA + paid stacks (ecosystem — not included in OSS)

**ALMA** is **MIT open source** (`pip install alma-memory`).  
It does **not** include proprietary products. Paid material must never ship for free inside ALMA.

---

## Who needs what?

| You are… | What you need |
|----------|----------------|
| **Developer, researcher, hobbyist, internal tool** — agents that learn, but you **don’t** ship production multi-agent apps to customers | **ALMA alone.** Full memory. Zero paid stack. |
| **Company shipping agentic software / ops in production** — you need audit, topology, witness, value proofs for customers, compliance, or security | **ALMA + paid accountability spine** (LAG · AWP · PayBotFin) as **separate products** |

**ALMA is always complete for memory.**  
**LAG + AWP + PayBotFin are optional accountability layers** — not memory features, not required for `retrieve` / `learn`.


## Hard rule (Chefe **1949**)

Graph Engineering **runtime** (ParviClaw Core + LAG + AWP + PayBotFin) must **never** be added to this OSS repository — not as code, not as a dependency, not as a submodule.  
ALMA documents the paid accountability suite for companies; it does **not** implement it.

## Product law (Chefe **1940**)

**LAG + AWP + PayBotFin** are the **perfect combo together** for **any agentic work accountability**.

| Piece | Accountability role |
|-------|---------------------|
| **LAG** | *What exists* — living architecture / fleet graph |
| **AWP** | *What happened* — tamper-evident witness of the act |
| **PayBotFin** | *What was allowed / valued* — authorize + numbered receipts ([paybotfin.com](https://paybotfin.com)) |

- **Companies / production agentic work:** this trio is the accountability spine.  
- **Developers (memory only):** not required — use **ALMA** alone.  
- **ALMA + this trio:** full story = *learn* (OSS) + *account* (paid).  
- Repos stay separate; **commercial suite** under PayBotFin brand is OK (**not** swallowing LAG into one monorepo — Chefe 1939).
- **Integrar (Chefe 1942)** = **seamless bridge/connect** (shared eng/leaf IDs, fail-soft, one ops story) — **same idea as ParviSight ↔ ParviClaw Core** — not monorepo absorb.

---

## What is **ALMA**? (OSS · this repo)

**Agent Learning Memory Architecture** — permanent memory that **learns**.

- Before work: retrieve what strategies worked or failed  
- After work: learn outcomes and anti-patterns  
- Your database; every AI tool  

**Job:** *smarter agents over time.*  
**Not:** a compliance ledger or architecture map of your fleet.

---

## What is **LAG**? (paid / private)

**LAG** = **Living Architecture Graph**.

A **paid** product that maps the **live agentic system**:

- **Who/what exists:** agents, skills, tools, workflows, connectors…  
- **How they connect:** edges (uses, produces, depends, guards…)  
- **Health over time:** orphans, broken links, topology  

Think of it as the **CMDB / topology map for AI agents** — not a chat log and not a memory of “tips.”

**Why it matters with ALMA:**  
ALMA answers *“what should we try next?”*  
LAG answers *“what is actually running and how is it wired?”*  

For a **company**, that map is how you govern fleets, spot drift, and show architecture truth.  
For a **solo developer** building a side project: you usually **don’t need** this.

---

## What is **AWP**? (witness protocol)

**AWP** = **Agent Witness Protocol**.

It turns agent **actions** into **tamper-evident receipts** (witness records) that can be **verified offline** — not “trust our log file.”

- Proves *an act happened* with integrity properties  
- Open verification patterns can exist for ecosystem trust  
- **Commercial multi-tenant issuance / production witness** is **not** bundled into ALMA  

**Why it matters with ALMA:**  
ALMA stores *lessons* (“blue-green worked”).  
AWP stores *evidence of the act* (“this run, this agent, this digest — verify it”).  

For a **company** under audit, customer trust, or security review: **accountability**.  
For a **developer** iterating on a laptop: **not necessary** for memory.

---

## What is **PayBotFin**? (paid / private)

**PayBotFin** is a **paid** product for **value and authorization** around agent acts.

- **Site / buy / pilot:** **[https://paybotfin.com](https://paybotfin.com)**  
- **Authorize / allow** high-stakes actions under policy  
- **Numbered commercial receipts** (meter, checkpoint, settle-style proofs)  
- Customer-facing **value accountability** — not just “the model said OK”  
- Related product witness namespace: [awp.paybotfin.com](https://awp.paybotfin.com/witness-record/v1)  

**Why it matters with ALMA:**  
ALMA learns *whether a strategy worked.*  
PayBotFin proves *whether the act was allowed and receipted* (money, policy, commercial proof).  

For a **company** selling or operating agentic services: **commercial + policy accountability**.  
For **developers and afins** who only want memory: **not necessary**.

---

## The combo: ALMA + LAG + AWP + PayBotFin = **auditable stack for companies**

```
                    ┌─────────────────────────────────────┐
                    │  ALMA (OSS) — learning memory       │
                    │  smarter decisions over time        │
                    └─────────────────────────────────────┘
                                      │
         optional paid accountability │ (companies / production)
                                      ▼
         ┌──────────────┬──────────────┬──────────────────┐
         │ LAG          │ AWP          │ PayBotFin        │
         │ map the      │ witness the  │ authorize /      │
         │ system       │ act          │ value receipt    │
         └──────────────┴──────────────┴──────────────────┘
```

| Layer | Question it answers | Audience |
|-------|---------------------|----------|
| **ALMA** | What should we remember and try next? | **Everyone** (devs → companies) |
| **LAG** | What exists and how is it connected? | **Companies** running fleets |
| **AWP** | Can we prove this act happened? | **Companies** needing audit/trust |
| **PayBotFin** | Was it authorized / receipted commercially? | **Companies** with value & policy |

### Why companies care about the full combo

1. **Memory without amnesia** (ALMA) — agents improve.  
2. **Architecture truth** (LAG) — you know what is deployed and linked.  
3. **Act integrity** (AWP) — acts are receipts, not editable log soup.  
4. **Value & policy** (PayBotFin) — high-stakes actions leave commercial proof.  

Together: an **auditable agentic ops stack** — learning **plus** accountability — without putting paid IP inside MIT ALMA.

### Why developers often stop at ALMA

- You want Claude/OpenClaw/scripts that **remember**  
- You are **not** delivering multi-agent **production** apps to customers  
- You do **not** need fleet topology, offline witness, or authorize receipts  

**ALMA alone is the right product.**  
LAG / AWP / PayBotFin are **extra security and accountability layers**, not “memory upgrades.”

---

## How they connect (when used together)

```
  Agent work
      │
      ├─ always:  ALMA.retrieve / ALMA.learn     ← memory (OSS)
      │
      └─ if production accountability (paid):
            LAG     → living map of the system
            AWP     → witness that the act happened
            PayBotFin → authorize / numbered value receipt
```

Private control planes (e.g. ParviClaw Core) may **call** ALMA as an OSS client.  
ALMA **never** depends on LAG or PayBotFin.

---

## What we will never do in this repo

- Depend on or vendor **LAG** or **PayBotFin**  
- Require paid APIs for `pip install alma-memory`  
- Market ALMA as “includes LAG/PayBotFin”  
- Force accountability products on developers who only need memory  

Commercial inquiries:

- **PayBotFin (paid accountability / value):** **[https://paybotfin.com](https://paybotfin.com)**  
- **Agentic Testari / Sentinel (security assess):** **https://agentictestari.com**

---

*Chefe 1932 · 1935 · 1937 · 1939 · OSS boundary 1927/1928*
