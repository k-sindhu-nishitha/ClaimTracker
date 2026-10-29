# Interview Guide: ClaimTrackr

## 1. The 30-second pitch
"I built a GenAI system that automates first-pass validation of medical insurance claims. The insurer's handbook is
chunked, embedded and stored in a FAISS vector index. When a claim comes in, the system extracts the diagnosis and amount
from the bill, runs deterministic checks, retrieves the relevant handbook pages, and has an LLM write a grounded report.
Rules and guardrails make the final decision, so the LLM can never approve something a rule rejected, and anything
uncertain goes to a human reviewer."

## 2. The 2-minute version (problem -> solution -> result)
- **Problem:** claim review is slow and error-prone because staff must read policy documents and bills by hand.
- **Approach:** RAG instead of fine-tuning, because policies change and I need answers that cite a handbook page.
- **Key design choice:** hybrid. Rules for what must be exactly right (amounts, names, exclusion list); LLM for
  reading unstructured bills and reasoning over policy text. The LLM output is validated and can only make a decision stricter.
- **Result:** a Flask app with a step-by-step decision trace, a JSON API, SQLite audit history and 30 automated tests.
- **Honest scope:** it is a prototype on a sample handbook and 3 sample bills; I have not measured production accuracy.

## 3. Map your resume bullet to the code
> "Built an end-to-end GenAI claim verification system using RAG, FAISS, and LLMs to automate medical insurance claim
> validation, including document retrieval, contextual reasoning, and backend processing workflows."

| Phrase | Where it lives |
|---|---|
| RAG | `processor.retrieve()` fetches excerpts, `llm.assess()` reasons over only those |
| FAISS | `rag.py` `VectorIndex` (`IndexFlatIP`, saved/loaded from disk) |
| LLMs | `llm.py` `OpenAIBackend`: bill extraction + assessment, both returning JSON |
| document retrieval | 3 queries per claim (required documents, exclusions, this diagnosis), de-duplicated, with file + page |
| contextual reasoning | `prompts.ASSESS_SYSTEM`: INFORMATION and EXCLUSION criteria checked against retrieved text |
| backend processing workflows | `processor.process()` 7-step pipeline, Flask routes, `/api/verify`, SQLite `storage.py` |

## 4. Walk through one claim (use the HIV bill)
1. Upload `HIV.pdf`, name Surbhit, amount 3000. `app.py` validates the form and that the file is really a PDF.
2. `pdf_utils` extracts text. `llm.extract_bill` returns `{patient: Surbhit, disease: HIV, expense: 3150}`.
3. `rules`: amount ok (3000 <= 3150), name ok, **exclusion screen fails**: "HIV" ~ "HIV/AIDS" (cosine 0.71 > 0.4).
4. `processor.retrieve`: query embeddings -> FAISS top-4 per query -> the General Exclusions page of the handbook comes back.
5. `llm.assess` writes the report using those excerpts, told that a rule already failed.
6. Guardrails: a failed rule always means REJECTED. The page shows the trace, the report and the cited handbook pages.
7. The decision is saved to SQLite and is visible under History.

Tip: run `python -m claimtrackr.cli demo -v` to show this trace in the terminal.

## 5. Design decisions and the reasoning behind them
- **RAG over fine-tuning:** policies change; RAG updates by re-indexing, and every answer can cite a page. Fine-tuning needs training data and cannot cite.
- **Chunk size 1000 / overlap 200:** large enough to keep a whole clause, overlap so a clause cut at a boundary appears whole once. Chunks are per page so each carries a page number for citations.
- **FAISS `IndexFlatIP` on normalised vectors:** exact cosine search. With ~130 chunks, approximate indexes add complexity with no benefit. For millions of chunks I would move to IVF or HNSW.
- **Three retrieval queries:** one question per concern (documents, exclusions, this diagnosis) retrieves better than one blended query.
- **Query rewriting:** I phrased queries like the handbook ("the Scheme shall not cover...") because that matches how the text is written. In the offline embedder it moved the real exclusions page from rank 4-5 to rank 1-2.
- **Structured JSON output + validation:** a free-text answer cannot be trusted by code. An invalid verdict means NEEDS_REVIEW.
- **Guardrails:** rules win over the LLM; an LLM rejection without a stated exclusion goes to a human; LLM outage fails safe.
- **Fail-safe default:** the cost of a wrong auto-approval is higher than the cost of sending a claim to a person.

## 6. Problems I found in my first version (great "what would you improve" answer)
1. The index was rebuilt on every request, so each claim paid for embedding the whole handbook twice. Fixed: build once, persist, reload, fingerprint to detect changes.
2. `claimed == billed` fell through to "invalid receipt". Fixed, with tests.
3. The exclusion screen used word overlap, so any "... disease" matched "Parkinson's disease". Fixed by ignoring generic words.
4. `int(expense)` crashed on values like "3,150.00". Fixed with a robust parser.
5. The report was rendered with `| safe`, which allows HTML injection from LLM output. Fixed.
6. The API key sat in a config file inside the repo. Moved to environment variables.

## 7. Questions you should expect

**Q: How do you reduce hallucination?**
Four layers: the LLM only sees retrieved excerpts and is told to use nothing else; structured output is validated; hard facts (amounts, names, exclusion list) are code, not LLM; and unclear cases go to a human.

**Q: Why not just paste the whole handbook into the prompt?**
It is 50+ pages: costly, slower, and models use long context less reliably. Retrieval sends only the few relevant pages and gives page-level citations.

**Q: Why FAISS and not Pinecone/Chroma?**
Small, static corpus, so a local library is enough: no service to run, no cost, easy to persist. A managed vector DB makes sense with many tenants, frequent updates, filters or huge scale.

**Q: How would you evaluate it?**
Build a labelled set of claims (approve/reject/review) and measure: verdict precision and recall (especially wrongly approved claims), retrieval hit-rate (does the right clause appear in the top-k?), and the share sent to humans. I have 5 smoke-test cases and 30 tests, but no accuracy figure yet, and I would not quote one.

**Q: What if the LLM is wrong?**
It cannot approve over a failed rule, and low-trust outputs (invalid JSON, rejection without a cited exclusion, incomplete information) route to NEEDS_REVIEW. The trace and cited pages let a human verify quickly.

**Q: What about prompt injection in an uploaded bill?**
Bill text and form fields are marked as untrusted data in the prompt, and the output is constrained to a JSON schema. Rules still run in code regardless of what the bill says. I would add more adversarial tests.

**Q: Sensitive health data?**
This prototype uses fictional data. In production: access control, encryption, audit logs (the SQLite history is a start), data-retention rules, and a decision on whether to use a self-hosted or contractually covered model. `OPENAI_BASE_URL` makes switching easy.

**Q: How would it scale?**
Move processing to a queue with workers (Celery), store claims in PostgreSQL, keep the index in memory per worker, cache embeddings, add retries and timeouts around LLM calls, and use IVF/HNSW if the corpus grows.

**Q: What are the embeddings?**
With a key, OpenAI `text-embedding-3-small`. Offline, a hashed TF-IDF fallback so the project runs anywhere; it is lexical, not semantic, which is why I also built the real path.

**Q: Why is exclusion screening not just LLM?**
A hard list is cheaper, instant, deterministic and auditable. The LLM covers what the list misses (synonyms like "senile dementia").

**Q: Biggest limitation?**
No labelled evaluation set, no OCR for scanned bills, and policy limits/waiting periods are not yet modelled as rules.

**Q: What would you do next?**
OCR for scans, an evaluation harness with retrieval metrics, a reranker, rules for benefit limits and waiting periods, async processing, and human-review UI with feedback that feeds the evaluation set.

**Q: Did LangChain play a role?**
The first version used LangChain's loader, splitter and FAISS wrapper. In this version I implemented those pieces directly (about 150 lines) to control the behaviour and to understand each step: `chunking.py` is the splitter, `rag.py` the vector store. If asked, say you know both and chose the direct version for transparency and fewer dependency changes.

## 8. Things to be careful about
- Know every file before the interview. Read in this order: `processor.py` -> `rules.py` -> `rag.py` -> `llm.py` -> `app.py`.
- Say clearly what runs offline versus with a real LLM; do not present the offline template as an AI model.
- Do not quote an accuracy number you have not measured.
- If you describe the first version versus this one, be accurate about which parts you wrote and which changed.
