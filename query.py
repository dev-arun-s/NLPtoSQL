"""
query.py — Convert natural language to SQL using your local schema index.

Usage:
    # Single query via command line
    python query.py "Show all customers with overdue invoices"

    # Interactive mode
    python query.py
"""

import json
import re
import os
import sys
from collections import defaultdict
from config import STORE_DIR, TOP_K
from llm_client import call_llm

CHUNKS_FILE = os.path.join(STORE_DIR, "chunks.json")
TFIDF_FILE  = os.path.join(STORE_DIR, "tfidf_index.json")

_chunks = None
_index  = None


def _load():
    """Load index files into memory (once per session)."""
    global _chunks, _index

    if _chunks is not None:
        return  # Already loaded

    if not os.path.exists(CHUNKS_FILE) or not os.path.exists(TFIDF_FILE):
        print("[ERROR] Index files not found. Please run ingest.py first:")
        print("    python ingest.py")
        sys.exit(1)

    with open(CHUNKS_FILE, encoding="utf-8") as f:
        _chunks = json.load(f)

    with open(TFIDF_FILE, encoding="utf-8") as f:
        _index = json.load(f)

    print(f"Loaded {len(_chunks)} tables from index.")


def tokenize(text: str) -> list:
    """Lowercase alphanumeric tokens, 3+ chars."""
    tokens = re.findall(r'[a-z0-9_]+', text.lower())
    return [t for t in tokens if len(t) >= 3]


def retrieve_schema(user_query: str, top_k: int = TOP_K) -> tuple:
    """
    Find the most relevant tables for the user query using TF-IDF + keyword boost.
    Returns (schema_context_string, list_of_matched_table_names).
    """
    _load()

    query_tokens = tokenize(user_query)
    scores       = defaultdict(float)

    # TF-IDF scoring
    for term in query_tokens:
        if term in _index:
            for doc_id_str, score in _index[term].items():
                scores[int(doc_id_str)] += score

    # Keyword boost: exact table or column name appearing in the query
    query_lower = user_query.lower()
    for i, chunk in enumerate(_chunks):
        if chunk["table_name"] in query_lower:
            scores[i] += 5.0                          # strong boost for table name match
        for col in chunk["columns"]:
            if len(col) > 3 and col in query_lower:   # skip short/generic column names
                scores[i] += 1.0
        for fk in chunk["foreign_keys"]:
            if fk in query_lower:
                scores[i] += 2.0                      # FK match = likely related table

    # Pick top-K
    ranked   = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    selected = [_chunks[i] for i, _ in ranked if scores[i] > 0]

    if not selected:
        # No scored matches — fall back to first top_k tables (shouldn't happen often)
        selected = _chunks[:top_k]

    table_names    = [c["table_name"] for c in selected]
    schema_context = "\n\n".join(
        f"-- Table: {c['table_name']} (from {c['source_file']})\n{c['ddl']}"
        for c in selected
    )
    return schema_context, table_names


def generate_sql(user_query: str, schema_context: str) -> str:
    """Send schema + question to the LLM and return the SQL."""

    system_prompt = """You are a SQL expert assistant working with an enterprise database.

Given a database schema (DDL) and a natural language request:
- Return ONLY the SQL query — no explanations, no markdown code fences, no comments before the SQL
- Use explicit JOINs (never implicit comma joins)
- Use short table aliases (e.g. o for orders, c for customers)
- Handle NULLs appropriately with COALESCE or IS NULL checks
- Use standard ANSI SQL unless the DDL implies a specific dialect (e.g. T-SQL, PL/SQL)
- If the provided schema is insufficient to answer the request, respond with exactly:
  -- Cannot determine: <brief reason>"""

    user_prompt = f"""DATABASE SCHEMA:
{schema_context}

USER REQUEST:
{user_query}

SQL:"""

    return call_llm(system_prompt, user_prompt)


def text_to_sql(user_query: str) -> dict:
    """Full pipeline: query → retrieve schema → generate SQL."""
    schema_context, matched_tables = retrieve_schema(user_query)
    sql = generate_sql(user_query, schema_context)
    return {
        "query":          user_query,
        "matched_tables": matched_tables,
        "sql":            sql
    }


def interactive_mode():
    """Run an interactive REPL for repeated queries."""
    _load()
    print("\n" + "=" * 60)
    print("  Text-to-SQL  |  type 'exit' to quit")
    print("=" * 60)

    while True:
        try:
            user_query = input("\nEnter your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_query:
            continue
        if user_query.lower() in ("exit", "quit", "q"):
            print("Goodbye.")
            break

        print("\nSearching schema...")
        try:
            result = text_to_sql(user_query)
            print(f"\nMatched tables : {', '.join(result['matched_tables'])}")
            print(f"\nGenerated SQL  :\n{result['sql']}")
            print("\n" + "-" * 60)
        except Exception as e:
            print(f"[ERROR] {e}")


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Query passed as command-line argument
        user_query = " ".join(sys.argv[1:])
        print(f"\nQuery: {user_query}")
        print("Searching schema...")
        result = text_to_sql(user_query)
        print(f"\nMatched tables : {', '.join(result['matched_tables'])}")
        print(f"\nGenerated SQL  :\n{result['sql']}")
    else:
        # No argument — run interactive mode
        interactive_mode()
