# Local Text-to-SQL Agents

[![CI](https://github.com/nazaryansurenn-amb/local-text-to-sql-agents/actions/workflows/ci.yml/badge.svg)](https://github.com/nazaryansurenn-amb/local-text-to-sql-agents/actions/workflows/ci.yml)

An offline analytics assistant for Microsoft SQL Server:

- A local LLM turns questions into read-only T-SQL.
- A team of specialist agents reviews and explains the results.
- A ChromaDB knowledge base adds domain context.

No data leaves the machine.

Case study: the operational database of irrigation Water User Associations in Armenia, covering
water users, deliveries and payments.

## Architecture

| Component | Implementation |
|---|---|
| UI | Streamlit (`app.py`) |
| LLM | Qwen (`qwen/qwen3.5-9b`), served by LM Studio's OpenAI-compatible API at `127.0.0.1:1234` |
| Database | SQL Server through pyodbc, Windows authentication, read-only queries only |
| SQL guard | `sql_safety.py`, see [SQL safety](#sql-safety) |
| Retrieval | ChromaDB in `chroma_db/`, `all-MiniLM-L6-v2` embeddings loaded offline from `models/`, 1,000-character chunks with 200 overlap. It indexes `.md`, `.txt` and `.pdf` files from `knowledge_base/` |
| Agents | `agents.py`: nine specialist prompts plus a coordinator |
| Memory | `memory_engine.py`: solved-task history and a metadata map of the SQL Server |

## How a question is answered

1. **Scope.** Either one or more selected tables (table mode), or no table (global mode).
2. **Context.** The model receives:
   - in table mode, the schema and up to 25 sample rows;
   - in global mode, the SQL Server metadata map;
   - in both modes, the knowledge chunks closest to the question.
3. **Draft.** Qwen writes T-SQL.
4. **Guard.** Every query passes `validate_read_only_sql` before it runs. Agent passes run at most
   3 queries, and each gets a `TOP 5000` cap unless it already limits its rows.
5. **Repair.** In the table Q&A, a query that fails goes back to the model once, together with the
   SQL Server error and the table's real column list.
6. **Answer.** Qwen writes the final answer from the rows the query actually returned, not from the
   draft.

## Agents

| Agent | Role |
|---|---|
| Senior Analytics Lead | Frames the question, KPIs and business meaning |
| Data Profiler | Structure, types, missing values, duplicates |
| Data Model / Join Discovery | Likely keys, join paths, table grain |
| SQL Analyst | Read-only T-SQL drafts |
| Anomaly Detection | Outliers, mismatches, trend breaks |
| Operations Analytics | Reads results as service usage, collection and data-quality KPIs |
| Power BI | Report pages, visuals, DAX |
| RAG Knowledge | Connects retrieved documents to the analysis |
| Validator | Checks drafts for unsafe SQL, invented columns and unsupported assumptions |

- **Team review (table mode):** runs eight of the agents in a fixed order. RAG Knowledge is left
  out. A coordinator prompt then merges their reports into one plan.
- **Global mode:** chooses specialists by keyword matching on the question, for example "join"
  selects Data Model and "anomaly" selects Anomaly Detection. The Validator always runs. The
  results are merged the same way.
- **Reports:** saved as Markdown, SQL and JSON in `outputs/agent_reports/`.

## Memory

- **Task memory:** each solved question with its table, SQL, execution summary and final answer.
  Stored in `outputs/agent_memory/`.
- **SQL Server metadata map:** a read-only scan of every user database. It records database,
  table and column names, types, estimated row counts, key-like columns and detected business
  concepts, and no row data.
- **Metadata search:** matches on concepts, so "paying users" also finds tables about payment,
  collection, contracts and account codes.

## SQL safety

T-SQL does not need a semicolon between statements: `SELECT 1 SHUTDOWN` runs a `SELECT` and then a
`SHUTDOWN`. So `sql_safety.py` does more than check that a query starts with `SELECT`:

- It reads comments, string literals and quoted identifiers the way SQL Server does, including
  nested comments and doubled quotes.
- It then rejects every reserved keyword that can start another statement (`SHUTDOWN`, `KILL`,
  `WAITFOR`, `DECLARE`, `SET`, ...).
- It also rejects `SELECT ... INTO`, the `OPENROWSET`, `OPENQUERY` and `OPENDATASOURCE` functions,
  and `xp_` / `sp_` procedures.

`tests/test_sql_safety.py` covers 46 allowed and blocked queries.

This is a filter, not a security boundary. The app connects with Windows authentication, so it has
the SQL Server rights of the Windows account running it. For hard read-only access, run it under an
account whose login has only `db_datareader`.

## Other features

- Table profiling: types, missing values, categorical distributions, automatic KPIs.
- Charts, including full-table charts computed as aggregate SQL instead of loaded rows.
- A Power BI export layer: CSV extracts, DAX measures, Power Query scripts and SQL view scripts.
  It is turned off by default. Set `SHOW_ADVANCED_BI_TRAINING = True` in `app.py` to enable it.

## Running

Requirements:

- Windows, with a local SQL Server and the `SQL Server` ODBC driver. The server name is `SERVER`
  in `app.py`.
- LM Studio serving a Qwen model on port 1234. The model name is `MODEL_NAME` in `app.py`.
- The `all-MiniLM-L6-v2` embedding model, downloaded once into `models/all-MiniLM-L6-v2/` for
  offline use.

```bash
pip install -r requirements.txt
streamlit run app.py        # or run_app.bat
pytest -q
```

Generated files go to `outputs/` and `chroma_db/`. Both are excluded from git.

## Status

- Prototype. Single user, runs locally.
- Windows-specific: ODBC driver name, Windows authentication and the launcher.
- `app.py` is about 4,600 lines, with UI and logic in one file.
- Agent selection in global mode uses keyword rules, not the model.
- Only `sql_safety.py` has automated tests. The LLM paths are exercised by hand.

## Stack

Python · Streamlit · pandas · pyodbc · SQL Server · LM Studio (Qwen) · OpenAI client · ChromaDB ·
sentence-transformers · matplotlib · pytest
