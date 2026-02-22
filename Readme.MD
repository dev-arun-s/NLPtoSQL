# Text-to-SQL Tool

Convert natural language questions into SQL queries using your enterprise DDL schema.

## Requirements

Only two packages needed:

```bash
pip install sqlparse        # DDL parsing (optional but recommended)
# urllib, json, re, math    # all Python stdlib — already available
```

> `sqlparse` is optional. If unavailable, the tool falls back to regex-based splitting.

## Setup

### Step 1 — Configure `config.py`

Edit these three values:

```python
DDL_FOLDER   = r"C:\DDL"                                          # your DDL folder
LLM_ENDPOINT = "http://your-internal-llm/v1/chat/completions"     # your endpoint
LLM_BEARER   = "your-bearer-token-here"                           # your token
LLM_MODEL    = "gpt-4"                                            # model name
```

Or set them as environment variables:

```bash
set LLM_ENDPOINT=http://your-internal-llm/v1/chat/completions
set LLM_BEARER=your-token
set LLM_MODEL=gpt-4
```

### Step 2 — Run ingest.py (once)

```bash
python ingest.py
```

This scans all `.sql` files in `C:\DDL`, parses every `CREATE TABLE` statement,
and saves two files:

```
schema_store/
├── chunks.json       ← all table DDL + metadata
└── tfidf_index.json  ← keyword search index
```

Re-run this only when your DDL files change.

### Step 3 — Query

```bash
# Single question
python query.py "Show all customers with overdue invoices"

# Interactive mode (keeps running, type exit to quit)
python query.py
```

## File Structure

```
project/
├── config.py          ← your settings
├── llm_client.py      ← HTTP call to your LLM endpoint
├── ingest.py          ← parse DDL files → build index
├── query.py           ← search index → call LLM → return SQL
└── schema_store/      ← auto-created by ingest.py
    ├── chunks.json
    └── tfidf_index.json
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `ingest.py` finds 0 tables | Check your .sql files contain `CREATE TABLE` statements |
| `query.py` says index not found | Run `ingest.py` first |
| LLM returns HTTP 401 | Check your bearer token in config.py |
| LLM returns HTTP 404 | Check your endpoint URL |
| Wrong tables matched | Try including the table name in your question |
