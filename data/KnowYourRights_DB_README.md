# Know-Your-Rights — Indian Legal RAG Database

A citation-ready vector database of **Indian central law**, built to power a public-facing
chatbot that explains ordinary people's legal rights in plain language **with citations**. It is a
retrieval substrate only — it does not generate answers. A downstream answer LLM (planned: NVIDIA
NIM + OpenAI Agents SDK with web tools) reads the retrieved sections and writes the user-facing
reply. This document is the complete handoff: what the database contains, how it was built, the
exact contracts for querying it, and the rules the answer layer must follow.

---

## 1. The bundle

`legal_db_bundle.zip` contains three artifacts:

| File | What it is |
|---|---|
| `legal_db/` | A **LanceDB** database directory. One table named **`laws`**. Holds every chunk's metadata, the 1024-dim embedding vector, and a BM25 full-text index over `embed_text`. |
| `chunks_metadata.parquet` | The same per-chunk rows as a pandas-friendly table **without** the vector column. Used at query time for candidate lookup, de-duplication, MMR, and to assemble the citation/answer context. |
| `enrichment_cache.json` | Per-section LLM enrichment keyed by `unit_id` → `{questions, keywords, category}`. Lets the build resume and lets you rebuild `embed_text` without re-calling the enrichment LLM. |

Current size: ~38,890 chunks (rows), 1024-dim vectors, ~263 MB zipped.

---

## 2. How it was built (pipeline)

`load → clean → currency-filter → map to one schema → chunk → enrich (LLM) → embed → LanceDB → evaluate`.

1. **Load** three sources (below) and verify shapes.
2. **Clean** whitespace, strip footnote/amendment markers (`4[(a)…]`), extract chapter headers, parse
   section numbers and act years.
3. **Currency filter:** drop the colonial criminal codes (IPC, CrPC, Indian Evidence Act) because
   they were repealed and replaced on **2024-07-01** by BNS / BNSS / BSA, which are added from CSVs.
   De-duplicate the Constitution (it appears both in a dedicated dataset and incidentally inside the
   bare-acts dataset; only the dedicated, properly-articled copy is kept).
4. **Map** every source to one unified schema (section 5).
5. **Chunk** by section (section 6).
6. **Enrich** each section once with an LLM (section 7).
7. **Embed** each chunk's `embed_text` with BAAI/bge-m3 and store in LanceDB with a BM25 index.
8. **Evaluate** with a small gold set + abstention/diversity stress probes (section 11).

The build runs on Kaggle (2×T4). The notebook is **load-or-build**: if this bundle is attached as a
Kaggle input it loads and skips straight to querying; otherwise it rebuilds from scratch.

---

## 3. Data sources

| Source | Kind | Approx. rows | Notes |
|---|---|---|---|
| `mratanusarkar/Indian-Laws` | Central acts (section-level) | ~34k → ~33.7k after filtering | ~1,000 acts. Repealed criminal codes dropped. Includes a **few incidental state acts** (see §9). |
| `Sharathhebbar24/Indian-Constitution` | Constitution articles | 454 + manual Preamble | Mapped as `Article N`. |
| BNS 2023 (CSV) | Criminal code | 358 | Bharatiya Nyaya Sanhita — replaces IPC. Effective 2024-07-01. |
| BNSS 2023 (CSV) | Criminal procedure | 531 | Bharatiya Nagarik Suraksha Sanhita — replaces CrPC. Effective 2024-07-01. |
| BSA 2023 (CSV) | Evidence | 170 | Bharatiya Sakshya Adhiniyam — replaces Evidence Act. Effective 2024-07-01. |

Statutory text is public-domain government material. The `mratanusarkar` dataset's license is not
explicitly stated — verify before any redistribution.

---

## 4. The unit model: section vs chunk

The **citation unit is the section** (an Act's section, or a constitutional Article). A `unit_id`
identifies it: `source_type|act_title|section_label`. Long sections are split into multiple **chunks**
for embedding; every chunk of a section carries the same metadata and the same enrichment. A
`chunk_id` is `unit_id#<n>`. Retrieval de-duplicates back to one row per section before reranking, so
the user never sees the same section twice.

---

## 5. Schema (columns in `laws` / `chunks_metadata.parquet`)

| Column | Meaning |
|---|---|
| `source_type` | `constitution` \| `central_act` \| `criminal_code` |
| `act_title` | Full title, e.g. `Right to Information Act, 2005` |
| `act_short_name` | Title with trailing year stripped |
| `act_year` | Year of enactment (int or null) |
| `act_number` | e.g. `45 of 2023` (sanhitas) or null |
| `section_label` | Section/article identifier as cited, e.g. `21`, `83`, `6A` |
| `section_num` | Numeric sort key derived from `section_label` |
| `section_name` | Marginal heading if available |
| `chapter` | Chapter/part name if available (Constitution: thematic part, e.g. "Fundamental Rights") |
| `jurisdiction` | `central` for all rows — **see the caveat in §9; trust `act_title`, not this field** |
| `status` | `in_force` \| `omitted` |
| `effective_date` | ISO date when known (sanhitas: `2024-07-01`) |
| `full_text` | The complete, untruncated section text (the source of truth to quote) |
| `source_snapshot` | Provenance + data vintage string (for "as-of" caveats) |
| `unit_id` | Section identity (see §4) |
| `chunk_id` | `unit_id#<n>` |
| `chunk_text` | The chunk slice of `full_text` |
| `category` | One of 18 citizen-facing categories (§7) |
| `citation` | Display string: `Article 21, Constitution of India` or `Section 6, Right to Information Act, 2005 (in force from …)` |
| `embed_text` | What was actually embedded (§7) — NOT what you show the user |
| `vector` | 1024-dim bge-m3 embedding, L2-normalised (LanceDB only) |

---

## 6. Chunking parameters

- Word-based split: `MAX_WORDS=480`, `OVERLAP=80` words.
- `MAX_CHUNKS=25` per section, so the few 80k-word schedules don't dominate the corpus.
- A section short enough is a single chunk. Enrichment is computed once per section and inherited.

---

## 7. Enrichment & `embed_text`

Each section was passed once to an LLM (`Qwen/Qwen2.5-7B-Instruct`, strict Pydantic-validated output)
to produce: a few **plain-language questions** a citizen might ask that this section answers, a list
of **keywords/synonyms**, and exactly one **category** from this taxonomy:

`Fundamental Rights, Criminal & Police, Consumer & Services, Employment & Labour, Family & Marriage,
Property & Housing, Women & Children, Privacy & Data, Health & Medicine, Education, Environment,
Taxation & Finance, Business & Companies, Information & RTI, Civil Procedure & Courts, Transport &
Motor, Government & Administration, Other`.

`embed_text` (the embedded string) = `act_title — Section label (section_name)` + the generated
questions + the keywords + the `chunk_text`. This is a **query-expansion trick**: embedding the
likely-questions alongside the legal text makes plain-language citizen queries match the formal
statutory language. **Show the user `full_text`/`citation`, never `embed_text`.**

---

## 8. Embedding contract (the part you must not get wrong)

The database is **tied to its embedder**. Queries must be encoded identically or retrieval silently
degrades.

- Model: **`BAAI/bge-m3`** (multilingual, XLM-RoBERTa backbone, 8192-token context, **1024-dim**).
- `max_seq_length = 1024` at build time (covers ~99% of chunks; ~1% are truncated for embedding only —
  `full_text` is always preserved in full).
- **No query prefix** (bge-m3 encodes queries and passages symmetrically). `EMBED_QUERY_PREFIX=""`.
- Vectors are **L2-normalised**; similarity is cosine / inner-product.

If you ever change the embedding model or dimension, you must re-embed the whole corpus.

---

## 9. Currency and jurisdiction — the two correctness rules

**Currency.** Each row carries `status`, `effective_date`, `act_year`, and `source_snapshot`, so the
answer LLM can say "this is the Act as it stood at <snapshot>." The colonial criminal codes are
already removed and replaced by BNS/BNSS/BSA (effective 2024-07-01). Anything enacted or amended
**after the snapshot** is out of scope for the data and must be checked LLM-side with web tools.

**Jurisdiction (important).** The corpus is *central* law, but `mratanusarkar` mixes in a **few state
acts** (e.g. the Maharashtra Rent Control Act), and the `jurisdiction` column labels everything
`central` — so that field is unreliable for those rows. **The `act_title` / `citation` is the
trustworthy jurisdiction signal** (a title beginning with a state name is a state law; "Delhi …" acts
are genuinely central because Parliament legislates for the Delhi UT). State coverage is incidental
and incomplete, not a real state-law corpus. The answer layer is responsible for jurisdiction, not
the database (see §12).

---

## 10. Known limitations

- Incidental, incomplete state-law coverage (above). Don't present it as authoritative state law.
- `jurisdiction` field is `central` for everything; rely on the title.
- ~1% of chunks exceed 1024 tokens and were truncated **for embedding only**.
- Post-snapshot amendments are not in the data — resolve them at answer time via web search.
- The eval gold set is small and illustrative (§11), not a benchmark.
- Source dataset license (`mratanusarkar`) is unstated.

---

## 11. Retrieval pipeline (the `search()` contract)

Given a user query string, retrieval does:

1. **Encode** the query with bge-m3 (no prefix), L2-normalised.
2. **Dense** search the `laws` table (top `FETCH_K=25`).
3. **Lexical** BM25 full-text search over `embed_text` (top `FETCH_K`). Falls back to dense-only if FTS
   is unavailable.
4. **Fuse** the two ranked lists with Reciprocal Rank Fusion (`RRF_K=60`).
5. **De-duplicate** to one row per `unit_id` (section).
6. **Rerank** the candidates with a cross-encoder and keep the top ~12.
7. **Diversify** the final ordering with MMR (`MMR_LAMBDA=0.6`) and return **`TOP_K=5`**.

Returned columns per hit: `citation, category, act_year, status, effective_date, score,
source_snapshot, full_text`.

**Reranker:** `Alibaba-NLP/gte-reranker-modernbert-base` (English, long-context), with automatic
fallback to `BAAI/bge-reranker-base`. The reranker is a **cross-encoder that reads text, not vectors**,
so it is independent of the embedder and can be swapped freely. Its tokenizer `max_length` must be
capped at the reranker's own position limit (e.g. 512 for the XLM-R fallback) — not the embedder's
1024 — or it throws a CUDA index error.

**Abstention signal:** if the top reranker `score` is below `LOW_SCORE=0.05`, treat the corpus as
having no good answer and let the answer LLM web-search instead of forcing a citation.

Eval snapshot on the gold set (after de-dup + corrected gold): **Recall@5 = 10/10**, with the answer
for arrest correctly resolving to BNSS (procedure), not BNS (offences).

---

## 12. The answer-layer contract (rules for the downstream LLM)

The database hands the LLM the right sections; the LLM must:

1. **Cite precisely.** Use the `citation` string and quote/paraphrase from `full_text`. Name the
   section, the Act, and the year. Never invent a citation.
2. **Check jurisdiction against the user.** Read the jurisdiction from the `act_title`/`citation`
   (not the `jurisdiction` field). If a hit is a state law and the user is elsewhere — or the user
   asked about a state subject the central corpus doesn't cover — say so plainly ("what I have is the
   Maharashtra Rent Control Act, which applies only in Maharashtra") and offer/perform a web search.
3. **Respect currency.** Frame answers as "as of `source_snapshot`," flag `status` (e.g. omitted),
   and verify anything that may have changed after the snapshot with web tools — especially recent
   amendments.
4. **Abstain gracefully.** If retrieval's top score is below the abstention threshold, don't force a
   statutory answer; explain that the topic looks like a state subject or isn't in the central corpus,
   and search.
5. **Plain language.** The audience is ordinary citizens, not lawyers. Explain, then cite.

---

## 13. Querying locally (minimal example)

```python
import lancedb, pandas as pd
from sentence_transformers import SentenceTransformer

DB_PATH, TABLE = "./legal_db", "laws"
emb = SentenceTransformer("BAAI/bge-m3", trust_remote_code=True)
emb.max_seq_length = 1024

chunks = pd.read_parquet("chunks_metadata.parquet")
db = lancedb.connect(DB_PATH)
tbl = db.open_table(TABLE)

def dense(query, k=25):
    qv = emb.encode([query], normalize_embeddings=True)[0]   # no prefix
    return tbl.search(qv).limit(k).to_pandas()

print(dense("how do I file an RTI request")[["citation", "category"]].head())
```

For the full hybrid + rerank + MMR `search()` (the §11 contract), reuse the retrieval cell from the
build notebook — it is identical in load mode and build mode. Reranking needs a GPU (or accept slower
CPU). The query embedder and reranker are the only models needed locally; no LLM is required just to
retrieve.

---

## 14. Reproducing or extending

- The build notebook regenerates this bundle end-to-end on Kaggle (2×T4, Internet On). It is
  load-or-build: attach this bundle to load; set `FORCE_BUILD=True` (or omit the bundle) to rebuild.
- Enrichment is cached by `unit_id`, so re-runs are fast — only new/changed sections call the LLM.
- Changing the **embedder** forces a full re-embed. Changing the **reranker** does not.
- To add a statute: append rows in the unified schema (§5), enrich the new sections, re-embed, rebuild.
