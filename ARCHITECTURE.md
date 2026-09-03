# How KnowYourRightsAI works

One question in, one cited answer out. This is everything in between — what runs, when, and
why it is there rather than something simpler.

---

## The whole pipeline

```mermaid
flowchart TD
    Q([" User question<br/><i>English · Hindi · Hinglish</i> "]) --> SAFETY

    SAFETY{{" 1 · SAFETY GATE<br/><small>patterns, then meaning — never a model call</small><br/><b>33/33 caught · 0/26 false alarms</b> "}}
    SAFETY -->|emergency detected| HELP[" Helpline card shown FIRST<br/><small>112 · 1091 · 1098 · 15100 · 14416</small> "]
    SAFETY --> LANG
    HELP --> LANG

    LANG[" 2 · LANGUAGE + VOCABULARY<br/><small>script & function words · acronyms · repeals</small><br/><b>IPC → BNS · RTI → Right to Information Act</b> "]
    LANG --> PLAN

    PLAN[" 3 · PLANNER <i>(fast model)</i><br/><small>intent · depth · sub-questions · which source answers what</small> "]
    PLAN -->|small talk| CHAT[" Concierge — streamed, no research "]
    PLAN -->|names a provision| EXACT[" 4a · EXACT LOOKUP<br/><small>no embedding, no reranking · 52 ms</small> "]
    PLAN -->|needs research| LOOP

    subgraph LOOP [" 4b · RESEARCH LOOP — up to 4 rounds, under a wall-clock deadline "]
        direction TB
        RW[" Query writer <i>(fast)</i><br/><small>2–4 phrasings per source</small> "] --> PAR
        PAR{{" run in parallel "}}
        PAR --> STAT[" legal_db<br/><small>the statute</small> "]
        PAR --> OFF[" official<br/><small>gov.in search</small> "]
        PAR --> WIKI[" wikipedia<br/><small>background only</small> "]
        STAT & OFF & WIKI --> GRADE[" Grader <i>(fast)</i><br/><small>drops vocabulary-only matches</small> "]
        GRADE --> CRAWL[" crawl4ai<br/><small>read pages · walk portals</small> "]
        CRAWL --> GAP{" Gap analyst <i>(fast)</i><br/>anything still missing? "}
        GAP -->|gaps + time left| RW
    end

    EXACT --> PROC
    LOOP --> PROC
    PROC[" 5 · PROCEDURE EXTRACTOR <i>(fast)</i><br/><small>steps · fee · deadline · appeal route · portal link</small> "]
    PROC --> PACK

    PACK[" 6 · CONTEXT PACKER<br/><small>token budget · statute always keeps a slot</small><br/><b>jurisdiction stamped on every statute</b> "]
    PACK --> WRITE

    WRITE[" 7 · WRITER <i>(large model)</i> — STREAMED<br/><small>shape follows the question · [S1] citation markers</small> "]
    WRITE --> FC

    FC{" 8 · SELF-CHECK <i>(fast)</i><br/><small>deep mode only</small><br/>anything here I should confirm? "}
    FC -->|confident| VERIFY
    FC -->|"fees · deadlines · thin support"| RECHECK[" Targeted web checks<br/><small>≤2 searches</small> "]
    RECHECK --> REWRITE[" Rewrite with what they said "]
    REWRITE --> VERIFY

    VERIFY[" 9 · CITATION VERIFIER<br/><small>code, not a model</small><br/><b>every [S1] must resolve — unresolvable ones are stripped</b> "]
    VERIFY --> OUT([" Answer + sources panel<br/><small>jurisdiction badges · trust tiers · verified count</small> "])
    CHAT --> OUT

    classDef gate fill:#fff4e6,stroke:#d97706,stroke-width:2px,color:#111
    classDef model fill:#e7f1ee,stroke:#1f6f5c,stroke-width:2px,color:#111
    classDef code fill:#eef2ff,stroke:#4f46e5,stroke-width:2px,color:#111
    classDef out fill:#f4f2ee,stroke:#6c665e,stroke-width:2px,color:#111
    classDef danger fill:#fdecea,stroke:#b3261e,stroke-width:2px,color:#111
    class SAFETY,GAP,FC gate
    class PLAN,RW,GRADE,PROC,WRITE,CHAT,REWRITE model
    class LANG,EXACT,PACK,VERIFY,CRAWL,RECHECK code
    class Q,OUT out
    class HELP danger
```

**Green = a model call. Blue = plain Python. Amber = a decision point. Red = safety.**

The shape of that diagram is the main design decision: **the blue boxes are in charge.** The
model never chooses to call a tool — it emits a validated plan, and Python executes it. A model
that cannot call tools cannot hallucinate a tool call, and a web page full of hostile
instructions cannot trigger one either.

---

## The safety gate — the one thing that runs before everything

It is first in the pipeline on purpose. Someone typing *"he is hitting me right now"* needs 112
before anything else happens, and that has to hold when every model provider is rate-limited,
retired, or down. So the gate never makes a model call.

```mermaid
flowchart LR
    M([" message "]) --> T1
    T1{{" TIER 1 · patterns<br/><small>~0 ms · no model, no network</small> "}}
    T1 -->|"strong match<br/><i>'hitting me' · 'I was raped'</i>"| FIRE
    T1 -->|"weak match<br/><i>bare noun: 'suicide'</i>"| GUARD
    T1 -->|no match| GUARD

    GUARD{" is this a question ABOUT the law?<br/><small><i>'punishment for…' · 'which act…' · '…laws in India'</i></small> "}
    GUARD -->|yes| PASS([" no card — continue to research "])
    GUARD -->|no| T2

    T2{{" TIER 2 · meaning<br/><small>cosine vs curated exemplars, bge-m3</small><br/><small>the embedding is reused by retrieval — free</small> "}}
    T2 -->|"≥ 0.64"| FIRE
    T2 -->|below| PASS

    FIRE[" HELPLINE CARD, before research<br/><small>112 · 1091 · 181 · 1098 · 15100 · 14416</small> "]

    classDef gate fill:#fff4e6,stroke:#d97706,stroke-width:2px,color:#111
    classDef danger fill:#fdecea,stroke:#b3261e,stroke-width:2px,color:#111
    classDef out fill:#f4f2ee,stroke:#6c665e,stroke-width:2px,color:#111
    class T1,T2,GUARD gate
    class FIRE danger
    class M,PASS out
```

**Why two tiers.** Patterns are instant and exact, and they cannot generalise. *"my partner
keeps hurting me and I am scared to go home"* matches nothing literal — and it is exactly what
someone types. So a second tier compares the message against curated exemplars of each crisis,
written the way frightened people write, in English, Hindi and Hinglish. It uses the embedder
already loaded for retrieval, and retrieval embeds the same string moments later and reads it
from cache, so the tier costs one embedding per turn rather than one per tier.

**Why strong and weak patterns.** Bare topic nouns are the vocabulary of the crime *and* of
every legal question about it. Matching them naively produced helpline cards on *"is suicide a
crime in India"* and *"child labour laws in India"*. Strong phrasings carry a subject or object
and fire unconditionally; weak ones are suppressed when the message reads as a question.

| | Disclosures caught | False alarms |
|---|---:|---:|
| Patterns only | 17 / 33 | 3 / 26 |
| **Both tiers** | **33 / 33** | **0 / 26** |

Scored on 33 labelled disclosures — literal and paraphrased, three languages — against 26 legal
questions deliberately chosen to be *about the same crimes in the same words*. Both sets live in
[`knowyourrights/safety_eval.py`](knowyourrights/safety_eval.py);
`python scripts/calibrate_safety.py` sweeps the threshold and prints the whole curve, weighting
a miss four times a false alarm.

**It fails open, never closed.** A broken embedder, or the `lite` profile which loads none,
degrades to tier 1 rather than to nothing.

---

## Retrieval, in detail

The statute search is where most of the engineering went.

```mermaid
flowchart LR
    Q([" query "]) --> EXP[" acronym expansion<br/><small>RTI → Right to Information Act, 2005</small> "]

    EXP --> EMB[" bge-m3<br/><small>local · fp16 · 61 ms</small> "]
    EXP --> BM[" BM25<br/><small>LanceDB · 23 ms</small> "]
    EMB --> ANN[" ANN vector search<br/><small>IVF-HNSW · <b>23 ms</b></small> "]

    ANN --> RRF{{" WEIGHTED FUSION (RRF) "}}
    BM --> RRF
    ACTF[" ×2.5 — searches limited<br/>to a named Act "] --> RRF
    GENF[" ×2.0 — searches limited to<br/>the Constitution + 2023 codes "] --> RRF

    RRF --> DED[" de-duplicate to one row per section<br/><small>keeping the best-ranked chunk</small> "]
    DED --> RR[" cross-encoder rerank<br/><small><b>309 ms</b> — 75% of the total</small> "]
    RR --> BOOST[" prefer general law over sectoral law<br/><small>Forest Act 0.994 vs BNSS 0.992 — the model cannot tell</small> "]
    BOOST --> MMR[" diversify on the <i>stored</i> vectors<br/><small>20 ms, and in the space actually searched</small> "]
    MMR --> TOP([" top sections<br/><b>399 ms end to end</b> "])

    classDef local fill:#e7f1ee,stroke:#1f6f5c,color:#111
    classDef fuse fill:#fff4e6,stroke:#d97706,stroke-width:2px,color:#111
    classDef step fill:#eef2ff,stroke:#4f46e5,color:#111
    class EMB,ANN,RR local
    class RRF,BOOST fuse
    class EXP,BM,DED,MMR step
```

Four of those boxes exist because of a specific observed failure:

| Box | What went wrong without it |
|---|---|
| **ANN vector search** | The corpus shipped with no vector index. Every query scanned 38,890 × 1024 floats — **159 MB**, 304 ms. |
| **Act-filtered lists ×2.5** | "How do I file an RTI" returned three unrelated *institute* Acts above the RTI Act. |
| **Prefer general law** | "Can the police arrest me" returned the **Navy Act, Forest Act and Railway Property Act** — all of which grant *someone* a power of arrest, in near-identical words. |
| **MMR on stored vectors** | Diversity was computed by re-encoding `chunk_text`, but the index holds vectors of `embed_text`. It was measuring in a space that was never searched. |

---

## Where the models come from

Two providers, because one turned out to be an unreliable dependency.

```mermaid
flowchart TB
    subgraph ROLES [" what each stage needs "]
        direction LR
        F[" FAST<br/><small>plan · queries · grade<br/>gaps · fact-check</small><br/><b>4–6 calls/question</b> "]
        W[" WRITER<br/><small>the answer</small><br/><b>1–2 calls/question</b> "]
        R[" RERANK<br/><small>retrieval</small> "]
    end

    F --> FR{" registry "}
    W --> WR{" registry "}
    R --> RR{" registry "}

    FR -->|1st| N1[" NVIDIA NIM<br/>nemotron-3-nano<br/><small>40 rpm <b>per model</b></small> "]
    FR -->|fallback| O1[" OpenRouter<br/>nemotron-3.5-lightning:free<br/><small>~20 rpm shared</small> "]

    WR -->|1st| N2[" NVIDIA NIM<br/>nemotron-3-super-120b<br/><small><b>2.8 s</b> measured</small> "]
    WR -->|fallback| O2[" OpenRouter<br/>gemma-4-31b:free · glm-5.2:free<br/>ultra-550b:free <small>(21.6 s — last resort)</small> "]

    RR --> LOC[" local cross-encoder<br/><small>every NIM rerank endpoint returns 410/404</small> "]

    classDef nim fill:#e7f1ee,stroke:#1f6f5c,stroke-width:2px,color:#111
    classDef or fill:#eef2ff,stroke:#4f46e5,stroke-width:2px,color:#111
    classDef local fill:#f4f2ee,stroke:#6c665e,stroke-width:2px,color:#111
    class N1,N2 nim
    class O1,O2 or
    class LOC local
```

**The order is measured, not assumed.** NVIDIA leads the fast role because its limit is 40/min
*per model* while OpenRouter shares ~20/min across all free models — and the fast role fires
4–6 times per question, so OpenRouter 429s almost immediately. NVIDIA also leads the writer
role on latency: 2.8 s against 4.1 s for the same model on OpenRouter.

And bigger is not better. `nemotron-3-ultra-550b` is the largest free model available anywhere
in this stack, and it took **21.6 seconds** for a two-sentence answer. It is kept last, as an
availability backstop.

**The embedder cannot move.** The corpus is embedded with `bge-m3`, so a different model means
re-embedding all 38,890 chunks. It runs locally, permanently, on whatever machine this is on.

### When a provider fails

```mermaid
flowchart LR
    C[" call "] --> S{" response "}
    S -->|200| OK([" answer<br/><small>clears the failure streak</small> "])
    S -->|429| P[" honour Retry-After<br/>countdown in the UI<br/>AIMD: rate ×0.7 "] --> C
    S -->|"410 / 404"| M[" sideline for this process<br/><small>3 failures in an hour to persist</small> "] --> NX[" next candidate<br/><small>often the other provider</small> "] --> C
    S -->|"5xx ×2"| NX
    S -->|"402 / 403"| NX
    S -->|deadline hit| DEG([" answer with what we have<br/><small>never nothing</small> "])

    classDef good fill:#e7f1ee,stroke:#1f6f5c,color:#111
    classDef bad fill:#fff4e6,stroke:#d97706,color:#111
    class OK,DEG good
    class P,M,NX bad
```

A single 410 no longer retires a model. NVIDIA returns them transiently under load — observed
live, a model 410'd on one call and answered normally on the next — and the earlier behaviour
of writing that verdict to disk meant one blip silently moved the whole app onto its fallback.

---

## What the user actually sees

```mermaid
flowchart LR
    O[" orchestrator "] -->|"typed SSE events"| UI[" browser "]

    subgraph EV [" the event stream "]
        direction TB
        E1[" <b>stage</b> — what is happening now "]
        E2[" <b>tool</b> — each search, with timing "]
        E3[" <b>source</b> — fills the panel <i>before</i> the prose starts "]
        E4[" <b>notice</b> — rate-limit countdown, jurisdiction warnings "]
        E5[" <b>procedure</b> — steps, fee, deadline, appeal "]
        E6[" <b>token</b> — the answer, streamed "]
        E7[" <b>sources_final</b> — the definitive citable set "]
        E8[" <b>verdict</b> — citations verified / stripped "]
    end
    O --- EV

    classDef ev fill:#f4f2ee,stroke:#6c665e,color:#111
    class E1,E2,E3,E4,E5,E6,E7,E8 ev
```

Typed events rather than a bare token stream, because a deep research turn runs for up to a
minute and a blank screen is indistinguishable from a crash. `sources_final` exists because the
packer re-assigns ids at the end — without it a citation chip could point at nothing.

---

## Jurisdiction — the thing it must not get wrong

Telling someone in Kerala that a Maharashtra rent law governs them is worse than telling them
nothing. So jurisdiction is never inferred:

```mermaid
flowchart TD
    S[" a statute section "] --> T{" read the Act's TITLE<br/><small>never the jurisdiction column</small> "}
    T -->|"starts with a state name"| ST[" <b>STATE</b><br/>applies only there "]
    T -->|"Constitution of India"| CO[" <b>CONSTITUTION</b><br/>applies nationwide "]
    T --> CE[" <b>CENTRAL</b><br/>applies across India "]

    ST --> CMP{" does it match<br/>the user's state? "}
    CMP -->|yes| FINE[" cite normally "]
    CMP -->|no| WARN[" red badge · warning line<br/>writer must say it does not apply "]
    CMP -->|state unknown| ASK[" say the answer depends on their state "]

    classDef warn fill:#fdecea,stroke:#b3261e,stroke-width:2px,color:#111
    classDef ok fill:#e7f1ee,stroke:#1f6f5c,color:#111
    class WARN warn
    class CE,CO,FINE ok
```

The corpus's own `jurisdiction` column says `central` for **every** row — including the 53
state Acts that leaked in from the source dataset. The Act title is the only trustworthy
signal, which is why it is the one that is used.

---

## What runs where

The same pipeline, sized to the machine. Only the two local models move; everything else is
identical, which is why **accuracy is a property of the system and latency is a property of the
box**.

| Profile | Embedder | Reranker | Recall@5 | Off-topic caught | Retrieval | Picked when |
|---|---|---|---:|---:|---:|---|
| `quality` | bge-m3 fp16 GPU | bge-reranker-v2-m3 | 100% | 11/11 | ~0.4 s | VRAM ≥ 3600 MiB |
| `balanced` | bge-m3 fp16 GPU | bge-reranker-base | 100% | 11/11 | ~0.4 s | VRAM ≥ 2900 MiB |
| **`cpu`** ← deployed | bge-m3 fp32 CPU | bge-reranker-base, pool 8 | **95.2%** | **10/11** | **6.6 s** | no usable CUDA |
| `cpu_lean` | bge-m3 fp32 CPU | none — fused RRF | 93% | 5/11 | 0.09 s | CPU box too slow to demo |
| `lite` | none | none — BM25 only | 90.5% | 5/11 | 0.06 s | under 2 GB RAM |

Two things this table is really saying:

**The cross-encoder's job is refusal, not ranking.** Removing it costs 7 points of Recall@5,
which is survivable, and takes off-topic rejection from 10/11 to 5/11, which is not. Without it
the answerable and off-topic score populations overlap, so no threshold separates them. The
deployed profile therefore keeps the model and shrinks its pool to 8 documents instead of
dropping it — 8 is enough to preserve the refusal and a third of the cost.

**Thresholds belong to a configuration, not a model.** They are keyed by backend, model,
quantisation and input length, because each of those moves the score distribution while leaving
the model's name unchanged. Reusing a calibration across them is silent and it breaks
abstention: measured, int8 scored against float32's threshold let off-topic questions through
6 times in 11 instead of 1.

### Measured on the deployment target

`m7i-flex.large`, 2 vCPU (**1 physical core**), 8 GB, Ubuntu 24.04, no GPU — against the
development laptop with an RTX 3050:

| | Laptop (GPU) | EC2 (CPU) | |
|---|---:|---:|---|
| Recall@5 | 95.2% | **95.2%** | identical |
| MRR | 0.861 | 0.854 | float32 vs fp16 flips near-ties |
| Off-topic wrongly answered | 0/8 | **0/8** | |
| Exact lookup | 4/4 | **4/4** | |
| Cold start | 56.7 s | **16.0 s** | no CUDA init |
| Dense search | 33 ms | **12 ms** | EC2 is faster |
| BM25 | 32 ms | **8 ms** | EC2 is faster |
| Cross-encoder | 309 ms* | **6,625 ms** | one core, no GPU |

\* on a GPU at full clock. Regenerate any of this on your own machine with
`python scripts/benchmark.py --all` then `python scripts/deploy_report.py`.

The searches feeding the reranker are *faster* on the small cloud box than on the laptop. All
of the difference is the cross-encoder, and all of that is having one physical core.

---

## Cost and limits at a glance

| | |
|---|---|
| Embedder | local, always — corpus-locked to bge-m3 |
| LLM calls per question | 4 (quick) · 8 (standard) · 20 (deep), as ceilings |
| NVIDIA free tier | ~1,000 credits, 40 rpm **per model** |
| OpenRouter free tier | 1,000 requests/day, ~20 rpm **shared across all free models** |
| Corpus | 38,890 chunks · 35,170 sections · 1,020 Acts · ~300 MB |
| Hosting | $0.0958/hour running, $1.60/month stopped |

Both free tiers are tracked client-side. OpenRouter's daily count survives restarts, so
bouncing the server cannot quietly blow the allowance, and a spent provider is skipped rather
than tried and refused.
