Session Log: Streamlit Dashboard and Self-Learning RAG Write-Back

Date: June 2026
Project: aml_rag_pipeline

Overview

This session covered two major additions to the AML RAG pipeline: building the
Streamlit dashboard for the project, and implementing the self-learning
write-back loop that lets the RAG knowledge base grow from LLM-classified
transactions over time. It also covers the bugs encountered along the way and
how each was resolved, since several of them are useful talking points for
explaining the engineering process in interviews.


Part 1: Initial Streamlit Dashboard

Built dashboard/app.py as a three-tab Streamlit app reading directly from
db/aml_pipeline.db (table transaction_results). Tabs: Overview, Transaction
Explorer, Pipeline Performance.

Challenges and fixes:

- rule_triggered exact-match bug. The Pipeline Performance tab initially
  showed "Skipped by Rule Engine: 0" because the code checked for an exact
  string match ("self_transfer") when the real stored value was a
  descriptive string ("Self-transfer: same account and bank"). Fixed by
  switching to a substring match using str.contains().

- General confirmation that the dashboard correctly reflects pipeline output:
  Overview metrics (2,500 total, 1,821 clean, 679 flagged) matched the
  original main.py run exactly, confirming the dashboard and database are in
  sync.


Part 2: UI Redesign

Redesigned the dashboard to use a sidebar navigation pattern (radio-based,
since Streamlit's native multipage feature splits across files) with
card-style metrics, alert-style transaction cards, and a purple/indigo accent
theme, inspired by a reference AML monitoring dashboard design.

Additions:
- Sidebar: project title, page navigation, Cost Assumptions (moved out of the
  floating widget into a collapsed expander), an LLM-only simulation toggle,
  a Reload Data button (clears Streamlit cache only, does not re-run the
  pipeline), an About section, and a GitHub repo link.
- Overview: card-style KPIs, custom risk-distribution bar list, typology bar
  chart.
- Transaction Explorer: filters plus a data table (card view was prototyped
  then removed per feedback; table-only is cleaner for large result sets).
- Pipeline Performance: new section reading from the batch_runs table
  (see Part 3) to show per-batch RAG vs LLM share and cumulative cost
  savings versus a simulated "LLM for every transaction" baseline.

Follow-up UI fixes after first review:
- Bar charts were too tall and visually heavy; fixed by setting explicit
  height and increasing bargap on all Plotly charts.
- Sidebar was too wide; constrained with min-width/max-width CSS.
- Removed the "Recent High-Risk Transactions" card section from Overview
  (was creating awkward empty space).
- Removed the Cards view toggle from Transaction Explorer, keeping the table
  view only, per feedback that it was unnecessary.

Open item: the "LLM-only mode (simulated)" toggle only recalculates cost
figures on the Pipeline Performance page. It does not trigger any real
pipeline run. There is currently no button in the dashboard that actually
re-runs main.py. This was flagged explicitly and is the subject of the
"Run Next Batch" discussion in Part 4.


Part 3: Self-Learning Write-Back Loop

This was the most significant functional addition. Previously, when the LLM
classified a novel transaction, the result was logged to SQLite but never
fed back into ChromaDB, so the RAG layer could never learn from it.

Changes made:

1. Added add_to_knowledge_base() to pipeline/rag_engine.py. It embeds the
   LLM's typology classification and reasoning using the same model and
   document format as the original seed typologies in knowledge_base/loader.py,
   and writes it to the same ChromaDB collection with a "source":
   "llm_discovered" metadata tag. Gating logic: skips write-back on
   LLM_ERROR status, on UNKNOWN typology, or on missing reasoning, so only
   genuine classifications get learned.

2. Wired the call into main.py, right after the LLM escalation block, so
   every successful LLM classification has a chance to be written back
   before the next transaction is processed.

3. Verified with an isolated test script (test_writeback.py, later deleted)
   feeding a synthetic transaction through analyze_with_llm() and
   add_to_knowledge_base() directly. Confirmed the ChromaDB collection count
   increased from 8 to 9 after a successful classification, and stayed at 8
   when the LLM returned UNKNOWN -- proving the gating logic works correctly
   in both directions.

Challenges and fixes:

- Initial test returned typology: UNKNOWN even for a transaction written to
  closely resemble the "Stack" typology. Root cause: the LLM prompt in
  llm_handler.py never told the model what the known typologies actually
  were -- it just said "respond with the closest typology or UNKNOWN" with
  no menu to choose from. Fixed by adding a KNOWN_TYPOLOGIES constant listing
  the typology names and short descriptions, and referencing it directly in
  the prompt. After the fix, the LLM correctly classified test transactions
  as Stack, Fan-Out Micro, and Cycle with no UNKNOWN results.

- The longer, more detailed typology-description version of the prompt
  significantly increased token usage per LLM call, which became relevant
  when Groq's free-tier daily token cap (100,000 tokens/day for
  llama-3.3-70b-versatile) was hit during testing. Resolved by trimming the
  KNOWN_TYPOLOGIES descriptions down to one concise line per typology
  instead of full paragraphs, cutting token usage substantially while
  keeping classification quality.


Part 4: Batch Run Logging and the 10-Run Demo Plan

Goal: demonstrate, using real (not scripted) pipeline behavior, that LLM
usage drops and RAG resolution increases as the knowledge base learns from
escalated transactions. Originally discussed as a fixed sequence of
percentages (100% LLM -> 50/50 -> 80/20 -> 95/5), but revised to let the
actual percentages emerge naturally from real batch runs rather than being
pre-scripted, since forcing exact percentages would not be an honest
reflection of real RAG/LLM behavior. Batch size was reduced from 500 to 200
transactions per batch to allow more, faster-turnaround runs (targeting
around 10 runs instead of 5).

Database changes (db/storage.py):
- Added a new batch_runs table: batch_number, total_transactions,
  rag_resolved, llm_escalated, llm_unknown, knowledge_base_size,
  run_timestamp.
- Added save_batch_run() and get_batch_runs().
- Note: since init_db() uses CREATE TABLE IF NOT EXISTS, adding a new column
  to an already-created table required manually dropping the old batch_runs
  table once so init_db() could recreate it with the updated schema.
  CREATE TABLE IF NOT EXISTS does not alter existing tables.

Pipeline changes (main.py):
- Batch size changed from 500 to 200 rows (load_transactions(chunksize=200)).
- LLM_ESCALATION_LIMIT changed from a global cap of 3 (effectively
  disabling escalation across the whole pipeline after 3 calls total) to a
  per-batch cap that resets at the start of every batch loop iteration.
  Initially set to 50, later reduced to 15 per batch as a safety margin
  against Groq's daily token limit, especially once batches escalate
  multiple times each.
- Added a 1-second time.sleep() between LLM calls as a precaution against
  Groq's 30 requests-per-minute cap.
- Added per-batch counters (batch_total, batch_rag_resolved,
  batch_llm_escalated, batch_llm_unknown) and a call to save_batch_run() at
  the end of each batch using collection.count() from rag_engine.py for the
  live knowledge base size.

Bug found and fixed: rate-limited (failed) LLM calls were initially being
miscounted as batch_llm_unknown, since the code only checked
llm_result["typology"] == "UNKNOWN" without checking llm_result["status"]
first. analyze_with_llm()'s exception handler returns typology: "UNKNOWN" as
a fallback on any API failure, which is a different thing from the LLM
genuinely classifying a transaction as unknown. Fixed by checking
status == "LLM_ERROR" first and excluding those from the unknown count, so
the "Genuinely Novel (Unclassified)" KPI reflects real model behavior, not
infrastructure failures.

Real-world constraint hit during testing: Groq's free-tier daily token cap
(100,000 tokens/day) was reached partway through testing, caused by the
cumulative token usage across multiple earlier test runs that day (write-back
tests, prompt tests, and partial batch runs), not by any single run alone.
This blocked completing a full clean 10-batch demo run within the same day.

Decision: rather than wait for the quota to reset, paused the live pipeline
testing and used the session to rebuild the dashboard UI instead, since the
dashboard's batch-trend visualizations could be built and verified against
the real (if partial) batch_runs data already collected (3 successful
batches), and the full clean 10-batch demo recording was deferred to a
future session once Groq's daily quota resets.


Open Items / Next Steps

1. Run a full, clean 10-batch demo sequence once Groq's daily token quota
   resets. Before running: clear transaction_results and batch_runs, and
   reset ChromaDB to the 8 seed typologies via knowledge_base/loader.py.

2. Build a real "Run Next Batch" button in the dashboard. Currently the
   dashboard only reads existing database data; there is no way to trigger a
   new pipeline run from the UI. This requires:
   - Modifying main.py to accept a specific batch number as a command-line
     argument (e.g. python main.py 4) rather than always starting from
     batch 1, which in turn requires checking whether pipeline/ingestor.py
     can skip directly to a given batch without reprocessing earlier rows.
   - The dashboard querying get_batch_runs() to determine the next batch
     number automatically.
   - Calling main.py as a subprocess from Streamlit with a progress
     indicator, then clearing the cache and reloading once it completes.

3. Build a real Reset button in the dashboard that performs the full reset
   sequence (clear both tables, reset ChromaDB) rather than requiring manual
   terminal commands.

4. Decide on a fallback model strategy for when Groq's daily quota is hit
   mid-demo: documented decision is to keep llama-3.3-70b-versatile as the
   primary model, with llama-3.1-8b-instant (separate quota bucket, 500K
   tokens/day) as a documented fallback rather than creating additional
   Groq accounts to bypass the limit.

5. Consider whether "rag_resolved" in batch logging should be split into
   rule-based resolutions versus true RAG vector-match resolutions for a
   finer-grained breakdown, since the current implementation counts
   everything that did not escalate to the LLM as "RAG resolved."
