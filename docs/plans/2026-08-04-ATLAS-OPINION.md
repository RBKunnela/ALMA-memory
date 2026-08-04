# Parecer — convite neoneye / Agent Memory Atlas (Chefe 556)

**Msg:** convite com 3 links  
**Data:** 2026-08-04  
**Orion**

## O que é o convite

Pessoa (neoneye) rodou **Claude Opus 5** sobre o repo **RBKunnela/ALMA-memory** e publicou no **Agent Memory Atlas** — atlas de **~140 sistemas de memória de agentes**, lidos em **código em commit pinado** (não só marketing).

| Link | Conteúdo |
|------|----------|
| [ALMA no atlas](https://neoneye.github.io/agent-memory-atlas/systems/alma-memory/) | Relatório profundo do ALMA (commit `164d2e3e…`, analisado 2026-08-04) |
| [Compare](https://neoneye.github.io/agent-memory-atlas/compare/) | Síntese do campo (tombstones, scope, trust, etc.) |
| [Discord](https://neoneye.github.io/agent-memory-atlas/discord.html) | Convite devs de memory systems → `discord.gg/6JjmzzuVHj` |

## Qualidade da análise (minha leitura)

**Alta.** Não é post de hype: é revisão de código com rubrica de 7 mecanismos, honestidade sobre “not found ≠ not needed”, e near-misses bem apontados.

### O que o atlas **elogia** no ALMA (e eu concordo)
1. **Tabela de anti-patterns** com `pattern` + **`why_bad`** + **`better_alternative`** — raro e valioso; o relatório diz que é o motivo principal de ler o ALMA.
2. **Scope enforced de verdade** (`WHERE project_id = ?` nas leituras) — marca ganha no sentido estrito.
3. **Vocabulário de verificação** (VERIFIED / UNCERTAIN / CONTRADICTED / UNVERIFIABLE + método) — melhor que a maioria.
4. **Retorno em buckets** (contradicted separado) — interface honesta para o agente consumidor.
5. Posicionamento: memória de **outcomes/heurísticas**, não só “user likes dark mode”.

### O que o atlas **critica** (e eu também prioritizaria no roadmap)
1. **VerificationStatus é calculado no retrieve e não é persistido** — contradição redescoberta a cada call; não vira fila de review nem query.
2. **Anti-patterns são guidance, não tombstone na write path** — não impedem o store de “re-aprender” o mesmo erro.
3. **Schema Postgres + SQLite em dois lugares** sem teste de paridade — risco de drift.
4. **ForgettingEngine poda sem audit trail** do que sumiu.
5. **LICENSE file** ausente no tree naquele commit (só MIT no pyproject) — detalhe legal a corrigir se for verdade ainda.
6. Benchmarks/ no repo **não foram executados** pelo analista — o R@5=0.964 do README é claim do projeto, não validado nesta leitura.

## Comparação (página compare) — encaixe do ALMA no campo

O campo inteiro sofre de:
- correção fraca (quase ninguém tem tombstone de valor rejeitado);
- trust como float, não estado;
- poucos testes de evidência negativa.

ALMA está **acima da média** em anti-patterns + scope + linguagem epistêmica, e **ainda não** no “estado de confiança persistido / write-guard”. Ou seja: **bom produto com gap claro e acionável** — não é “pior que Mem0 no marketing”; é “diferente e com 1–2 fios de ouro ainda soltos”.

## Brownfield local (início)

Clone: `/opt/pedro/projects/ALMA-memory`  
README: pip install, MCP, multi-agent, LongMemEval claim, dual storage.

## Parecer sobre o **Discord / convite**

| | |
|--|--|
| **Quem** | Rede de **developers de memory systems**, não spam aleatório |
| **Valor** | Peer review, visibilidade do ALMA, ideias de roadmap (tombstone, persist verification) |
| **Risco** | Baixo se só participar; médio se prometer features sem gate; não compartilhar secrets/prod |
| **Recomendação** | **Sim, vale entrar** (você ou Maia como face produto; Orion/Pedro técnicos se precisar). Tratar o relatório do atlas como **input de produto**, não como ataque. |

### Resposta sugerida (curta, EN)

> Thanks — strong read of the repo. The anti-pattern columns and scope-as-predicate are intentional; the verification-not-persisted and write-path anti-pattern points are fair and on our radar. Happy to join the Discord. Best, Renata

## O que **não** fazer
- Não reescrever o ALMA só para “passar” a rubrica do atlas sem produto.
- Não misturar isso com cyber-eval Q-Factory (outro trilho).
- Não assumir que o commit analisado = main de hoje (revalidar LICENSE + verification).

## Próximos (se Chefe quiser)
1. Entrar no Discord  
2. Issue/ADR: persist `VerificationStatus` + optional write-guard from anti_patterns  
3. LICENSE file no root  
4. Teste de paridade Postgres/SQLite  

— Orion  
