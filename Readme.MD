# Text-to-SQL Tool

Convert natural language questions into SQL queries using your enterprise database schema. No vector database, no external ML models — just Python, a local file index, and your internal LLM endpoint.

---

## How It Works

```
C:\DDL\*.sql files
       │
       ▼
  [ ingest.py ]
       │  Parses every CREATE TABLE statement
       │  Builds a TF-IDF keyword search index
       ▼
  schema_store/
  ├── chunks.json        ← table DDL + metadata
  └── tfidf_index.json   ← keyword search index
       │
       ▼
  [ app.py / query.py ]
       │  User types a natural language question
       │  Top matching tables retrieved via TF-IDF scoring
       │  Schema context + question sent to your LLM endpoint
       ▼
  SQL query returned
```

---

## Project Structure

```
project/
├── config.py              ← your settings (endpoint, token, paths)
├── llm_client.py          ← HTTP call to your LLM endpoint
├── ingest.py              ← parse DDL files and build local index
├── query.py               ← command-line query interface
├── app.py                 ← Flask web UI backend
├── templates/
│   └── index.html         ← web UI frontend
└── schema_store/          ← auto-created by ingest.py
    ├── chunks.json
    └── tfidf_index.json
```

---

## Requirements

```bash
pip install flask sqlparse
```

- **flask** — web UI server
- **sqlparse** — DDL parsing (optional; falls back to regex if unavailable)
- **urllib, json, re, math** — Python stdlib, no install needed

No vector database. No ML model downloads. No other dependencies.

---

## Setup

### Step 1 — Configure `config.py`

Open `config.py` and set these values:

```python
DDL_FOLDER   = r"C:\DDL"                                           # folder with your .sql files
LLM_ENDPOINT = "http://your-internal-llm/v1/chat/completions"      # your LLM endpoint
LLM_BEARER   = "your-bearer-token-here"                            # your bearer token
LLM_MODEL    = "gpt-4"                                             # model name your endpoint expects
TOP_K        = 12                                                   # number of tables passed to LLM
```

Or set them as environment variables (takes precedence over config.py):

```bash
set LLM_ENDPOINT=http://your-internal-llm/v1/chat/completions
set LLM_BEARER=your-token-here
set LLM_MODEL=gpt-4
```

### Step 2 — Ingest your DDL files (run once)

```bash
python ingest.py
```

This recursively scans `C:\DDL` for all `.sql` files, parses every `CREATE TABLE` statement, and saves the index to `schema_store/`. You will see per-file progress:

```
========================================
  Text-to-SQL Ingestion
========================================
Found 312 .sql files in C:\DDL
Parsing files...
  [1/312] orders.sql        → 1 table(s)
  [2/312] customers.sql     → 1 table(s)
  ...
Total tables parsed: 298
Building search index...
Saving to schema_store/...
  chunks.json      — 298 tables (842 KB)
  tfidf_index.json — 4231 terms (310 KB)
✓ Ingestion complete.
========================================
```

> Re-run `ingest.py` only when your DDL files change.

### Step 3 — Use the tool

**Option A — Web UI (recommended)**

```bash
python app.py
```

Then open **http://localhost:5000** in your browser.

**Option B — Command line**

```bash
# Single question
python query.py "Show all customers with overdue invoices"

# Interactive mode
python query.py
```

---

## Using the Web UI

| Feature | Description |
|---|---|
| Type your question | Plain English — no SQL knowledge needed |
| **Ctrl + Enter** | Keyboard shortcut to run the query |
| Matched tables | Shows which tables were used to generate the SQL |
| Copy button | Copies the raw SQL to clipboard |
| Recent Queries | Last 8 queries shown at the bottom — click to reload |

---

## How the Schema Search Works

The tool does **not** use embeddings or a vector database. Instead it uses a pure-Python TF-IDF keyword index:

1. Every table's DDL, column names, and FK references are tokenized and scored
2. Table names are weighted 5×, FK references 2×, column names 1×
3. When you ask a question, the top 12 most relevant tables are selected
4. Those DDL definitions are sent to your LLM as context alongside your question

This approach is fast (~10ms for 300 tables), requires no ML infrastructure, and works well for enterprise schema sizes.

---

## What Gets Stored Locally

| File | Contents | Typical size (300 tables) |
|---|---|---|
| `schema_store/chunks.json` | Full DDL + table name, columns, FK refs, source file | ~1–3 MB |
| `schema_store/tfidf_index.json` | Term → document scores inverted index | ~500 KB |

Both files are plain JSON — readable in any text editor. Total footprint for 300 tables is under 4 MB.

---

## LLM Endpoint

The tool calls any **OpenAI-compatible endpoint** (`POST /v1/chat/completions`) with Bearer token authentication. This includes:

| Platform | Typical URL |
|---|---|
| Azure OpenAI | `https://<resource>.openai.azure.com/openai/deployments/<model>/chat/completions?api-version=2024-02-01` |
| Internal gateway | `http://your-internal-llm/v1/chat/completions` |
| Ollama (local) | `http://localhost:11434/v1/chat/completions` |
| vLLM | `http://internal-server:8000/v1/chat/completions` |

To use a different request/response format, edit `_build_payload()` and `_parse_response()` in `llm_client.py`.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ingest.py` finds 0 tables | Verify your `.sql` files contain `CREATE TABLE` statements |
| `ingest.py` shows no output | Check that `C:\DDL` exists and contains `.sql` files |
| `query.py` says index not found | Run `ingest.py` first |
| Button does nothing in browser | Open browser DevTools (F12) → Console tab and check for errors |
| LLM returns HTTP 401 | Bearer token is wrong — check `LLM_BEARER` in config.py |
| LLM returns HTTP 404 | Endpoint URL is wrong — check `LLM_ENDPOINT` in config.py |
| Wrong tables matched | Include the table name explicitly in your question |
| Foreign keys show schema prefix (e.g. `abc`) instead of table name | Ensure `ingest.py` uses the regex: `r'REFERENCES\s+(?:\[?\w+\]?\.)?\[?(\w+)\]?'` |

---

## Re-ingesting After DDL Changes

```bash
del schema_store\chunks.json
del schema_store\tfidf_index.json
python ingest.py
```

Or on Linux/Mac:
```bash
rm schema_store/chunks.json schema_store/tfidf_index.json
python ingest.py
```
