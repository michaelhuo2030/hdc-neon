# Where HDC Helps (and Where It Doesn't)

A living document on the practical sweet spot of Hyperdimensional Computing (HDC) — where it offers real structural advantages over conventional vector retrieval (HNSW, FAISS, pgvector), and where it doesn't.

This is shaped by ongoing conversations with practitioners, customers, and HN/Reddit/Twitter commenters. Credits at the bottom.

---

## TL;DR

**HDC shines when your workload has any of these properties:**

1. **Compositional queries** — multiple conditions combined via algebra ("A bind B but not C, satisfies D")
2. **Streaming append** — real-time data flow where index rebuild kills HNSW
3. **Interpretability / audit trail** — each dimension is a known concept projection
4. **Cross-modal / cross-domain mapping** — bind operations naturally encode "concept X in domain A → concept Y in domain B"
5. **Privacy-sensitive domains** — linear ops give differential-privacy hooks for free
6. **Energy-constrained edge** — bit-packed, hardware-friendly, ~1000× lower per-op energy on ReRAM CIM

**HDC does NOT structurally beat HNSW/FAISS when:**

- Pure semantic similarity at moderate scale (N < 1M, single-query) — HNSW is already very fast
- High-precision continuous-value retrieval — HDC's 1-bit (or 2-4 bit) quantization loses fine gradients
- The compositional structure of your query doesn't matter

---

## Sweet-Spot Matrix

| Scenario | HDC advantage | HNSW limitation |
|---|---|---|
| Multi-condition compositional query ("bought A + not B + risk-tier C") | `bind` / `bundle` / `unbind` are O(1) algebraic ops | Multi-pass intersection, accuracy degrades with each pass |
| Streaming append (real-time transactions, agent memory) | O(1) append, no index restructure | Graph rebalance is O(log N); at N=1B+ becomes infeasible |
| Concept decomposition + audit ("which 3 principles produced this rule?") | Each dim is a known concept projection — fully decomposable | Black-box embedding, no interpretability |
| Cross-modal / cross-domain mapping ("philosophy concepts in this engineering problem") | `bind` operations cross-encode source × target domain | Single embedding space; needs domain-specific re-training |
| Privacy-sensitive workflows (finance, medical) | Linear-only ops + per-dim noise → natural DP guarantees | Black-box embedding can leak via inversion attacks |
| Edge / battery-constrained inference | 1-bit XOR popcount, 1.25 KB/doc | fp32 vectors, ~4-6 KB/doc + graph overhead |

---

## Three Worked Examples

### Example 1 — Banking marketing: the canonical compositional case

A bank wants to find customers who:
1. Viewed Fund X in the mobile app in the last 30 days but didn't buy
2. Behave like the historical 6-month-pre-churn cohort
3. Are compliance-eligible for the fund
4. Within the active marketing window

That's **4 simultaneous conditions over heterogeneous data sources** (transaction logs + behavior embeddings + compliance rules + temporal filter).

**Conventional RAG pipeline** does this as 4 sequential filter passes:
- SQL query → 30k candidates
- Embedding similarity vs churn cohort → 3k candidates
- Compliance rule pass → 300 candidates
- LLM final ranking → 30 candidates

Each step loses signal. At 5+ conditions, accuracy collapses. Business teams blame the LLM ("we need GPT-5"). They're wrong — the bottleneck is the retrieval infrastructure not supporting **algebraic composition**.

**HDC formulation:**
```
target_query =
    bind(viewed_fund_X, recent_30d) +    // saw it lately
    similarity_to(churn_pattern) +        // behaves like pre-churn
    bind(compliance_pass, fund_X) +       // legally eligible
    active_window
```
One pass, one Hamming-distance scan. No accuracy cliff. Auditable per-dimension.

Status today: this architecture is the **L4 layer** in a multi-tier knowledge architecture. Few production banks have it; most are stuck at L1 (relational) + L3 (vector). It's a 1-3 year window for whoever builds it first.

---

### Example 2 — Personal knowledge wiki: cross-concept mapping

Use case: you have ~1,000-10,000 notes spanning philosophy, engineering, business, and personal journal. You ask:

> "For this complex problem I'm facing, what philosophical concepts apply? What underlying patterns from prior work?"

Conventional semantic search returns "notes that *mention* X." That's not what you want — you want notes that **structurally** share principles with X, even if the surface vocabulary differs.

HDC fits because:

- **Concepts are vocabulary, not just text** — encode a vocabulary of 200-500 named concepts (道德经 principles, systems theory primitives, Bayesian inference patterns, design patterns) each as its own HV
- **Each note is a `bundle` of its concept-tags** — encoding "this note touches concepts A, C, F" is just `bundle(A, C, F)`
- **Cross-domain bridge** is automatic — a problem encoded as `bundle(uncertainty, decision_under_pressure, family)` retrieves prior notes that touch the same concept set, regardless of whether they were written about engineering, business, or your kid

Scale is small (10k notes), so HNSW would also be fast. But **HNSW would not surface "structural overlaps across domains"** — it would return surface-similar notes. HDC's `bind`/`bundle` algebra over a hand-crafted concept vocabulary is the right tool for the job.

Setup cost: ~3 hours to define a personal concept vocabulary (200-500 concepts), then automate the encoding for new notes via embedding model. Lifetime cost: ~$1 in API calls for the entire wiki, then $0/query.

---

### Example 3 — Agent episodic memory

An always-on agent (assistant, robot, AR glasses) needs to remember:
- Every interaction stored as it happens (streaming append)
- Retrievable by compositional query ("show me sessions where the user was frustrated AND we were debugging the same module AND I gave bad advice")
- At N=1M+ events over months
- On <0.1 W battery budget

HNSW + DRAM hits the memory wall (huge index + constant graph rebalancing). HDC + ReRAM CIM is a structural fit:
- O(1) append to a linear store
- Compositional retrieval via `bind`/`bundle`
- Hardware-friendly: 1.25 KB/event, bit-packed; analog-compute-friendly on ReRAM crossbar
- Energy: storing weights at crossbar cells + Ohm's-law multiply-accumulate ≈ 1000× lower per-op energy than DRAM ping-pong

This is exactly the workload our 28nm ReRAM CIM chip is being designed for. It's also what `hdc-neon` (the algorithm side) is prep work for.

---

## The Workflow in 60 Seconds

```
[ONE-TIME SETUP per corpus]
  Text/data items
    → Embedding model (Qwen3-Embedding / bge / nomic-embed)
    → dense float vector (384-2560 dim)
    → Random projection matrix · sign()
    → bipolar HDC vector (10K - 1.3M bits, bit-packed)
    → store on disk (10MB per 1k items)

[QUERY-TIME, per query]
  Natural-language query
    → same embedding model + same projection (must be reused)
    → HDC query vector
    → Hamming-distance search (hdc-neon: 2.98 ms/q at N=200K)
    → top-k IDs
    → (optional) feed top-k context to an LLM for natural-language synthesis

[COMPOSITIONAL QUERY (advanced)]
  query_hv = bind(concept_A_hv, concept_B_hv) + bundle(filter_C_hv, filter_D_hv) - exclude_E_hv
    → same Hamming search
    → results that satisfy the algebraic composition
```

**LLM is involved at two points: (1) encoding text → embedding (one-time per item), (2) optional final answer synthesis.** The HDC search itself is pure XOR + popcount — zero tokens.

---

## Embedding Model Choice: It Matters

HDC inherits the semantic quality of the embedding model. Garbage in → garbage out. Brilliant in → brilliance mostly preserved (by Johnson-Lindenstrauss + sign-preserving distance).

| Tier | Model | Dim | Notes |
|---|---|---|---|
| Free | nomic-embed-text | 768 | English-focused, local, fast |
| Free | bge-small-zh-v1.5 | 512 | Chinese, lightweight |
| Recommended | **Qwen3-Embedding-4B** | 2560 | Strong CN+EN, local-deployable (~16GB GPU) |
| Top-tier | gte-Qwen2-7B-instruct | 4096 | Slightly better, needs 32GB GPU |
| Top-tier API | text-embedding-3-large | 3072 | OpenAI, ~$50 per 10k items |

A non-obvious finding from our research (Paper #2 / sparse-T HDC): **at very high HDC dimensions (D ~1.3M), a weaker embedding model can sometimes match a stronger one** — because the high-D HDC space "spreads out" concepts the weaker model has compressed. Active research direction.

---

## When NOT to Use HDC (be honest)

- **N < 100k, single-query semantic search, no composition needed** → just use FAISS Flat or HNSW
- **Continuous-value retrieval where 0.1% precision matters** (e.g., scientific feature matching) → HDC bipolar loses too much
- **Mature application with HNSW infrastructure already shipped** → don't re-platform unless the compositional/streaming axis is a hard requirement
- **Pure ANN benchmark optimization (recall@k on standard test sets)** → HNSW + product quantization is the SOTA, HDC isn't trying to win this game

The honest framing: HDC isn't a *faster* HNSW. It's a *different shape* of retrieval, optimized for compositional algebra + streaming + interpretability + hardware co-design. If you don't need those properties, HNSW is the right answer.

---

## Open Questions (where research is active)

1. **Multi-bit HDC for real text encoders** — does 2-3 bit per dim materially help over bipolar on production embeddings? (Synthetic data says yes by +12pp R@5; real bge text says noise.)
2. **Optimal D for a given application** — D=10K vs D=100K vs D=1.3M trade-off curve.
3. **HDC + LLM hybrid retrieval** — using HDC as a coarse compositional filter, then LLM rerank.
4. **Cross-civilization concept alignment** (the Universal Wisdom Atlas direction) — can HDC + bilingual embeddings surface deep structural overlap between e.g. Confucian and Stoic ethics?

---

## Thanks To

This document grows from real conversations. If your question or insight shaped a section, you'll be credited here.

- *(slot for HN commenters whose questions sharpened the framing)*
- *(slot for X / LinkedIn / email contributors)*
- *(slot for fellow practitioners)*

If you have a use case where you're unsure HDC fits — open an issue at github.com/michaelhuo2030/hdc-neon/issues with the workload shape, or DM. We're collecting these honestly: signal in, signal out.

---

*Last updated: 2026-05-25. Maintained alongside `hdc-neon` and the 5-paper HDC research series. Apache 2.0 — copy, adapt, contribute.*
