# Evaluation Report

Everything below is measured, not estimated. Regenerate with:

```bash
python scripts/benchmark.py --all     # free — GPU only, no API calls
python scripts/benchmark.py --e2e     # full turns, spends API credits
```

Raw output lands in `.runtime/benchmark.json`.

**Test machine:** NVIDIA RTX 3050 Laptop (4 GB VRAM), 16 GB RAM, 8 physical cores, Windows 11.
Profile `balanced` — `BAAI/bge-m3` fp16 and `BAAI/bge-reranker-base` fp16, both on GPU.

---

## 1. What is in the database

| | |
|---|---|
| Chunks (embedded units) | **38,890** |
| Sections (citation units) | **35,170** |
| Distinct Acts | **1,020** |
| Act years covered | **1838 – 2023** |
| In force / omitted | 34,824 / 346 |
| Resident index at runtime | **12.6 MB** |
| On disk | ~350 MB |

**By source**

| Source | Sections | Note |
|---|---|---|
| Central Acts | 33,656 | ~1,000 statutes |
| Criminal codes | 1,059 | BNS, BNSS, BSA (2023) |
| Constitution | 455 | Articles + Preamble |

**By citizen-facing category** (assigned by an LLM during the build, one per section)

| Category | Sections | | Category | Sections |
|---|---:|---|---|---:|
| Government & Administration | 13,552 | | Transport & Motor | 1,102 |
| Business & Companies | 4,193 | | Health & Medicine | 1,029 |
| Criminal & Police | 2,745 | | Consumer & Services | 767 |
| Other | 2,293 | | Family & Marriage | 633 |
| Education | 2,086 | | Environment | 366 |
| Property & Housing | 1,756 | | Women & Children | 320 |
| Civil Procedure & Courts | 1,421 | | Information & RTI | 264 |
| Taxation & Finance | 1,331 | | Privacy & Data | 90 |
| Employment & Labour | 1,170 | | Fundamental Rights | 52 |

That distribution matters when reading the results: the corpus is heavily weighted toward
administrative statute, while the questions people actually ask cluster in the small
categories — Fundamental Rights is 52 sections out of 35,170.

### Two properties of the data that shaped the system

**Only 1,060 of 35,170 sections (3%) carry a marginal heading.** The 2023 criminal codes have
them; the Constitution and central Acts do not. This is why the reranker is given a heading
only when one exists — see §6.

**53 Acts are state legislation** that leaked in from the source dataset (Maharashtra Rent
Control Act and similar). They are detected by title prefix and flagged in the UI, because the
`jurisdiction` column says `central` for every row and cannot be trusted.

### What is deliberately absent

The Indian Penal Code, Code of Criminal Procedure and Indian Evidence Act were repealed on
2024-07-01 and removed from the corpus. Questions about them are redirected to the BNS / BNSS /
BSA with the substitution stated. Also absent, and reported as gaps rather than substituted:
the Prevention of Money Laundering Act and the Digital Personal Data Protection Act.

---

## 2. How it was evaluated

Three question sets, in [`knowyourrights/eval_data.py`](knowyourrights/eval_data.py).

### 2.1 Gold set — 42 questions

Written the way a citizen would ask, not the way a statute is worded, and spread across the
taxonomy. Each is graded on whether the expected Act appears in the returned citations.

| Category | n | Example |
|---|---:|---|
| Fundamental Rights | 10 | *"can I move the Supreme Court if my fundamental rights are violated"* |
| Criminal & Police | 8 | *"how long can police keep me in custody before producing me before a magistrate"* |
| Employment & Labour | 5 | *"sexual harassment at my workplace complaint committee"* |
| Consumer & Services | 3 | *"my consumer complaint against a defective product"* |
| Information & RTI | 3 | *"my RTI was rejected, how do I appeal"* |
| Women & Children | 3 | *"dowry demand by my in-laws"* |
| Family & Marriage | 2 | *"grounds for divorce under Hindu law"* |
| Transport & Motor | 2 | *"compensation for a road accident death"* |
| Health & Medicine | 2 | *"when can a woman legally terminate a pregnancy"* |
| Education, Property, Taxation, Environment | 1 each | *"builder delayed possession of my flat"* |

**Some questions accept more than one answer, and they must.** Indian law frequently gives two
correct citations:

- the 24-hour custody limit is in **both** Article 22 and the BNSS;
- the Code on Wages, 2019 **subsumed** the Minimum Wages Act, 1948 — both are in the corpus;
- "penalty for polluting the environment" is answered by the Environment (Protection) Act
  *and* by BNS §§279–280, which criminalise fouling water and vitiating the atmosphere.

That last one was originally scored as a **miss**. Reading what the system actually returned —
BNS §280, *"whoever voluntarily vitiates the atmosphere… shall be punished with fine"* — showed
the retrieval was right and the gold answer was too narrow. Widening it was correcting the test,
not moving the goalposts; the same treatment was applied to the other multi-answer cases.

### 2.2 Stress set — 11 questions it should *not* answer

| Kind | n | Expected behaviour |
|---|---:|---|
| State subjects | 3 | flag as state law, don't imply central coverage |
| Not legal at all | 8 | abstain — *"who won the cricket world cup in 2011"*, *"how do I fix a memory leak in my python program"* |

Eight off-topic questions rather than two, because two samples cannot locate a threshold.

### 2.3 Exact-lookup set — 4 questions

Questions naming a provision (*"what does Article 21 say"*, *"Section 35 BNSS"*). These must be
resolved by exact fetch, not similarity search — someone naming a section deserves that section.

---

## 3. Retrieval accuracy

| Metric | Result |
|---|---|
| **Recall@5** | **100.0%** (42/42) |
| Recall@3 | 95.2% (40/42) |
| Recall@1 | 78.6% (33/42) |
| **MRR** | **0.873** |
| Exact lookups correct | **4/4** |

Recall@10 is also 100%, i.e. nothing is merely ranked deep — everything correct is inside the
top 5.

### The same corpus under every configuration it can run in

The headline row needs a GPU. Most places this gets deployed do not have one, so every
CPU-viable configuration was measured on the same 42 questions rather than assumed:

| Configuration | Recall@5 | MRR | Top-1 | Off-topic caught | Retrieval |
|---|---:|---:|---:|---:|---:|
| **`cpu`/`balanced`, rerank pool 24** | **100%** | **0.873** | 79% | **11/11** | ~400 ms GPU · **20 s CPU** |
| `cpu`, rerank pool 12 | 95.2% | 0.849 | 76% | 10/11 | 0.57× the above |
| **`cpu`, rerank pool 8** ← deployed | 95.2% | **0.861** | 79% | 10/11 | **0.32×** |
| `cpu`, rerank pool 6 | 93% | 0.846 | 79% | 10/11 | 0.26× |
| `cpu_lean` — dense + BM25, no reranker | 93% | 0.787 | 71% | **5/11** | **71 ms** |
| `lite` — BM25 only, no models | 90.5% | 0.769 | 69% | **5/11** | **61 ms** |

Three things fall out of this table, none of which were obvious beforehand:

**Accuracy is device-independent.** The `cpu` profile scores *identically* to the GPU one —
100%, MRR 0.873, 4/4 exact, 11/11 stress — because it runs the same two models. Only latency
moves, and it moves by a factor of fifty.

**Pool 12 was strictly the wrong choice.** It had been the deployed setting. Pool 8 matches its
recall, beats its MRR (0.861 vs 0.849), restores top-1 to the full 79%, and runs nearly twice as
fast. Nothing was being bought with those extra four documents.

**The cross-encoder's real job is refusal, not ranking.** Dropping it costs 7 points of
Recall@5 — survivable. It also takes off-topic rejection from 10/11 to **5/11**, which is not:
calibration shows the answerable and off-topic score populations *overlap* once it is gone
(floor 0.414 against ceiling 0.468), so no threshold can separate them and the best achievable
cut is 90% accurate. Reranking just 8 candidates is enough to keep that defence, which is why
the deployed configuration keeps the model and shrinks the pool instead of the reverse.

**Per-category Recall@5 is 100% across all 13 represented categories.** With 42 questions the
per-category counts are small (1–10 each), so this says the system has no blind *domain*, not
that each category is proven to three decimal places.

### Against the notebook baseline

| | Notebook | Now | |
|---|---|---|---|
| Recall@5 | 95.2% | **100.0%** | +4.8 pts |
| MRR | 0.783 | **0.873** | +11.5% |
| Recall@1 | 66.7% | **78.6%** | +11.9 pts |
| Median retrieval latency | 674 ms | **399 ms** | −41% |

The gains came from four changes, each traceable to an observed failure (§6): weighted
multi-query fusion, preferring general law over sectoral law, MMR on the stored vectors rather
than re-encoded text, and adding a vector index.

---

## 4. Abstention — knowing when not to answer

For a legal tool this matters more than recall. A confident wrong citation is worse than none.

| | Result |
|---|---|
| Off-topic questions answered anyway | **0 / 8** |
| Answerable questions wrongly refused | 2 / 42 |
| State-subject questions flagged as state law | **3 / 3** |
| Calibrated threshold | `LOW_SCORE = 0.0832` |

The two refusals are questions where retrieval genuinely failed — its best hit scored below the
floor. Abstaining and searching the web is the correct behaviour there, not a defect.

**Why the threshold is calibrated rather than chosen.** The build notebook shipped `0.05`. It
was tuned for a different cross-encoder, and score distributions do not transfer between
rerankers. Inherited unchanged it let **1 in 2** off-topic questions through. Measuring it —
running answerable questions against unanswerable ones and placing the cut in the gap between
the two populations — took that to **0 in 8**.

Thresholds are stored per backend *and* model. Changing either invalidates them, which is why
`scripts/calibrate.py` is a required setup step rather than a tuning nicety.

---

## 5. Latency

### 5.1 Inside one retrieval

Median of 5 runs, `FETCH_K=25`, `RERANK_POOL=24`:

| Stage | Median | p90 | |
|---|---:|---:|---|
| Embed the query | 61 ms | 96 ms | bge-m3 fp16, one short string |
| Vector search | **23 ms** | 31 ms | was 304 ms before the index — §6 |
| BM25 search | 23 ms | 28 ms | LanceDB full-text |
| Fetch candidate rows | 27 ms | 30 ms | 24 rows by id |
| Fetch stored vectors | 20 ms | 22 ms | for MMR, instead of re-encoding |
| **Cross-encoder rerank (24 docs)** | **309 ms** | 311 ms | the dominant cost |
| **Whole search** | **399 ms** | 458 ms | |

Reranking is now ~75% of retrieval time. It is also what makes the results good, so it stays;
the tuning lever is `RERANK_POOL`.

Exact lookups bypass all of this: **52 ms median**, no embedding and no reranking.

### 5.2 A full answer, end to end

End-to-end time is dominated by the language model, not by this system. That became unavoidably
clear when NVIDIA's endpoint degraded mid-evaluation, so both conditions are reported:

| Depth | Provider healthy (~1 s/call) | Provider degraded (~2.6 s/call) | LLM calls |
|---|---|---|---|
| `quick` | **5–8 s** | 18–36 s | 3 |
| `standard` | **10–25 s** | 26–120 s | 4–5 |
| `deep` | **17–60 s** | 227 s | 8–9 |

Retrieval contributed **under half a second** in every one of those runs. The variance is
entirely provider-side — measured directly, a trivial `reply with OK` call went from ~1 s to
**2.6 s** between the two conditions.

Two consequences, both now handled:

- **Answers stream.** Time-to-first-token, not total time, is what a user experiences.
- **The writer can fail entirely.** During the degradation two turns produced no answer at all.
  That now falls back to a model-free digest of the provisions found — raw statutory text with
  citations. Worse than a written answer; far better than a blank screen.

### 5.3 Startup

Cold start is **40–140 s** depending on memory pressure: loading bge-m3 (~1090 MiB VRAM), the
reranker (~760 MiB), opening LanceDB, and a warmup pass. The warmup is deliberate — the first
CUDA encode costs 1.86 s and every one after it ~25 ms, so that cost is paid at boot rather
than by the first user. The server accepts connections immediately and reports `ready: false`
until warm.

### 5.4 Resource cost

| | |
|---|---|
| bge-m3 fp16 | 1090 MiB VRAM |
| bge-reranker-base fp16 | 760 MiB VRAM |
| Peak with both | ~2690 MiB of 4096 MiB, ~1.4 GB left free |
| Host RAM spike while loading | ~2.3 GB |
| Resident section index | 12.6 MB |

The load spike is the most likely way to wedge a small machine, so a RAM floor check refuses to
load below a threshold and retrieval starts keyword-only instead of crashing.

---

## 6. Changes that moved the numbers

Each of these came from watching an actual failure, not from theory.

### No vector index existed

`dense_search` measured **304 ms** — suspicious for 38,890 rows. The table had a BM25 index and
**no vector index**, so every query brute-force scanned 38,890 × 1024 float32 ≈ **159 MB**.

Building an `IVF_HNSW_SQ` index (64 partitions, cosine, ~42 MB, 23 s to build):

| | Before | After |
|---|---|---|
| Vector search | 304 ms | **23 ms** |
| Whole retrieval | 644 ms | **399 ms** |
| Recall@5 | unchanged | unchanged |
| MRR | 0.837 | **0.849** |

Approximate search cost nothing measurable in quality — exact nearest neighbours were still
returned in 12/12 self-retrieval checks. Reproducible via `scripts/build_index.py`.

### General law lost to sectoral law

Dozens of statutes grant *someone* a power of arrest — forest officers, naval authorities,
railway police — in wording nearly identical to the criminal procedure code's. The cross-encoder
scored the Essential Services Maintenance Act at **0.994** against the correct BNSS section at
**0.992**. It cannot tell them apart, because textually they are alike; the difference is which
statute governs the person asking.

Preferring the general codes for criminal/policing questions, applied to ranking *and* to MMR's
relevance signal, changed "what are my rights if the police arrest me" from returning the Navy
Act, Forest Act and Railway Property Act to returning five BNSS sections.

### Diversity was computed in the wrong vector space

MMR re-encoded each candidate's `chunk_text` to measure redundancy — but the index stores
vectors of `embed_text` (heading + generated questions + keywords + chunk). Diversity was being
measured in a space that was never searched. Reading the stored vectors back is both correct and
**20 ms instead of ~12 long-document encodes**.

### Section headings helped 3% and hurt the rest

Prepending each section's marginal heading for the reranker made results *worse* — the correct
Article 22 dropped out of the top 5. The data explains it: headings exist for 1,060 sections and
are empty for the other 34,110. For 97% of the corpus this was adding a bare citation line as
noise. Adding it only where a heading exists lifted BNS §318 ("Cheating") into the top 2 for
*"what is the punishment for cheating"* without costing anything elsewhere.

### Acronyms had to be expanded — but not for the reranker

"RTI" shares no tokens with "Right to Information Act, 2005", so a raw search misses. Expansion
fixes keyword and vector search. But the expanded string reads as broken English — *"how do I
file an Right to Information Act, 2005"* — and degrades a cross-encoder trained on natural
questions. Search gets the expanded text; the reranker gets the original.

---

## 7. Answer quality

Retrieval accuracy is measurable; answer quality is partly not. What *is* checked automatically:

**Citation integrity.** Every `[S1]` marker in an answer is resolved against the evidence the
writer was actually given. Unresolvable markers are stripped and reported. Across the
end-to-end runs: **11 citations verified, 0 unsupported**.

**Grounding by construction.** The writer receives only packed evidence and is instructed to
cite from it. Three structural defences back that up:

1. The pipeline is code-orchestrated — the model emits a validated plan, Python executes it. A
   tool call cannot be hallucinated, and crawled web text cannot trigger one.
2. An LLM grader drops sources that merely share vocabulary with the question.
3. Repealed codes are absent from the corpus, so they cannot be cited from it.

Defence 3 has a gap worth naming: the *model* still remembers the CrPC. Under weak retrieval it
once recommended it from memory. The writer is now explicitly told that reaching for the CrPC
means it is working from stale memory rather than its sources.

**Known weakness.** Relevance grading is a model call and varies between runs. It occasionally
over-rejects and thins an answer; once it rejected all six correct BNSS sections. A rescue path
covers total rejection — if grading rejects everything while retrieval was confident, the top
statutes are kept. Partial over-rejection remains the main open issue.

---

## 8. Test suite

73 tests, fully mocked — no GPU, no network, no API credits, ~100 s. They run on Python 3.11 and
3.12 in CI; 3.12 is what Ubuntu 24.04 ships, so it is the version the deployed box actually runs.

| Area | What is guarded |
|---|---|
| Rate limiting | 429 pauses and resumes; AIMD floor; deadline beats infinite retry; auth errors are not retried |
| Model failover | an unreachable model fails over; **a single transient 410 does not permanently retire a model**; repeated failures are recorded |
| Retrieval | weighted RRF; MMR prefers relevance but drops near-duplicates |
| Context packing | a statute always survives against a much larger web page; the token budget is respected |
| Citations | fabricated markers are stripped; the grader-rescue path fires only on confident retrieval |
| Safety | emergency phrasing is detected; *"how do I report domestic violence"* is correctly **not** an emergency |
| Vocabulary | acronyms expand once, not recursively; `Section 6 of the RTI Act` doesn't parse as section "6OF" |
| Language | English stays English (regression: Hindi *"the"* collides with English *"the"*) |
| Memory | stale citation markers are stripped from history |

---

## 9. Honest limitations

1. **42 gold questions is a working test set, not a benchmark.** It shows no blind domain; it
   does not establish per-category accuracy to any precision.
2. **The gold set is mine.** I wrote both the questions and the expected answers, which risks
   fitting the system's strengths. It is versioned in the repo so the bias is at least visible.
3. **Answer quality has no automated judge.** Citation integrity is verified; whether the prose
   is *useful* is assessed by reading it.
4. **Central law only.** State subjects are covered incidentally and flagged, not answered.
5. **Snapshot-bound.** Amendments after the corpus build are absent; deep mode web-checks but
   anything time-sensitive needs verifying.
6. **One machine, one run each.** Latency figures are medians from a single laptop, not a
   distribution across hardware.
7. **End-to-end latency is mostly not mine to control**, as the provider degradation made
   plain.
