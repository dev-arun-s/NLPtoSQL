"""
ingest.py — Parse all DDL files in DDL_FOLDER and build a local search index.

Run this once (or whenever your DDL files change):
    python ingest.py

Output:
    schema_store/chunks.json      — All table DDL + metadata
    schema_store/tfidf_index.json — Keyword search index
"""

import os
import sys
import json
import re
import math
from collections import defaultdict
from config import DDL_FOLDER, STORE_DIR

CHUNKS_FILE = os.path.join(STORE_DIR, "chunks.json")
TFIDF_FILE  = os.path.join(STORE_DIR, "tfidf_index.json")


# ── DDL Parsing ────────────────────────────────────────────────────────────────

def extract_columns(stmt: str) -> list:
    """Best-effort column name extraction from a CREATE TABLE statement."""
    skip_keywords = {
        "PRIMARY", "FOREIGN", "UNIQUE", "INDEX", "KEY",
        "CONSTRAINT", "CHECK", "CREATE", "TABLE", "WITH",
        "CLUSTERED", "NONCLUSTERED", "ALTER", "ADD"
    }
    cols = []
    # Find the content between the outermost parentheses
    match = re.search(r'\((.*)\)', stmt, re.DOTALL)
    if not match:
        return cols

    body = match.group(1)
    for line in body.splitlines():
        line = line.strip().rstrip(',')
        if not line:
            continue
        col_match = re.match(r'^(\[?(\w+)\]?)\s+\S+', line)
        if col_match:
            name = col_match.group(2)
            if name.upper() not in skip_keywords:
                cols.append(name.lower())
    return cols


def parse_file(filepath: str) -> list:
    """
    Parse a single .sql file and return a list of table chunks.
    Handles files with multiple CREATE TABLE statements.
    Does NOT require sqlparse — uses regex splitting as fallback.
    """
    try:
        with open(filepath, encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except Exception as e:
        print(f"    [ERROR] Could not read {filepath}: {e}")
        return []

    # Try sqlparse if available, else use regex split
    statements = []
    try:
        import sqlparse
        statements = [s.strip() for s in sqlparse.split(content) if s.strip()]
    except ImportError:
        # Fallback: split on GO or semicolons (common in enterprise SQL scripts)
        raw = re.split(r'\bGO\b|;', content, flags=re.IGNORECASE)
        statements = [s.strip() for s in raw if s.strip()]

    chunks = []
    for stmt in statements:
        # Only process CREATE TABLE statements
        if not re.search(r'\bCREATE\s+TABLE\b', stmt, re.IGNORECASE):
            continue

        # Extract table name — handles schema prefix like dbo.TableName or [dbo].[TableName]
        match = re.search(
            r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?'
            r'(?:\[?[\w]+\]?\.)?' 
            r'\[?(\w+)\]?',
            stmt, re.IGNORECASE
        )
        if not match:
            continue

        table_name = match.group(1).lower()
        columns    = extract_columns(stmt)
        fk_refs    = [f.lower() for f in re.findall(r'REFERENCES\s+(?:\[?\w+\]?\.)??\[?(\w+)\]?', stmt, re.IGNORECASE)]
        comments   = re.findall(r'--[^\n]*', stmt)

        chunks.append({
            "table_name":  table_name,
            "source_file": os.path.basename(filepath),
            "columns":     columns,
            "foreign_keys": fk_refs,
            "comments":    comments,
            "ddl":         stmt,
        })

    return chunks


def scan_all_ddl(folder: str) -> list:
    """Recursively walk folder and parse every .sql file."""
    if not os.path.exists(folder):
        print(f"\n[ERROR] DDL folder not found: {folder}")
        print("  Please update DDL_FOLDER in config.py and try again.")
        sys.exit(1)

    sql_files = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(".sql"):
                sql_files.append(os.path.join(root, f))

    if not sql_files:
        print(f"\n[ERROR] No .sql files found in: {folder}")
        sys.exit(1)

    print(f"Found {len(sql_files)} .sql files in {folder}")
    print("Parsing files...")

    all_chunks  = []
    empty_files = []

    for i, filepath in enumerate(sql_files, 1):
        chunks = parse_file(filepath)
        if chunks:
            all_chunks.extend(chunks)
            print(f"  [{i}/{len(sql_files)}] {os.path.basename(filepath)} → {len(chunks)} table(s)")
        else:
            empty_files.append(os.path.basename(filepath))
            print(f"  [{i}/{len(sql_files)}] {os.path.basename(filepath)} → (no CREATE TABLE found, skipped)")

    if empty_files:
        print(f"\nNote: {len(empty_files)} file(s) had no CREATE TABLE statements.")

    return all_chunks


# ── TF-IDF Index (pure Python stdlib) ─────────────────────────────────────────

def tokenize(text: str) -> list:
    """Lowercase alphanumeric tokens, 3+ chars."""
    tokens = re.findall(r'[a-z0-9_]+', text.lower())
    return [t for t in tokens if len(t) >= 3]


def chunk_search_text(chunk: dict) -> str:
    """
    Build the text that represents a chunk for indexing.
    Boost table name and columns by repeating them.
    """
    return " ".join([
        (chunk["table_name"] + " ") * 5,
        " ".join(chunk["columns"]) * 3,
        " ".join(chunk["foreign_keys"]) * 2,
        " ".join(chunk["comments"]),
        chunk["ddl"]
    ])


def build_tfidf_index(chunks: list) -> dict:
    """
    Build a TF-IDF inverted index.
    Returns: { term: { "doc_id": score, ... }, ... }
    """
    N      = len(chunks)
    corpus = [tokenize(chunk_search_text(c)) for c in chunks]

    # Document frequency: how many docs contain each term
    df = defaultdict(int)
    for doc_tokens in corpus:
        for term in set(doc_tokens):
            df[term] += 1

    # Build inverted index with TF-IDF scores
    index = defaultdict(dict)
    for doc_id, doc_tokens in enumerate(corpus):
        if not doc_tokens:
            continue
        tf = defaultdict(int)
        for t in doc_tokens:
            tf[t] += 1
        for term, freq in tf.items():
            idf   = math.log((N + 1) / (df[term] + 1)) + 1.0
            score = (freq / len(doc_tokens)) * idf
            index[term][str(doc_id)] = round(score, 6)

    return dict(index)


# ── Main ───────────────────────────────────────────────────────────────────────

def build_store():
    print("=" * 60)
    print("  Text-to-SQL Ingestion")
    print("=" * 60)

    # 1. Scan and parse
    chunks = scan_all_ddl(DDL_FOLDER)

    if not chunks:
        print("\n[ERROR] No tables were parsed. Check your DDL files.")
        sys.exit(1)

    print(f"\nTotal tables parsed: {len(chunks)}")

    # 2. Build TF-IDF index
    print("\nBuilding search index...")
    index = build_tfidf_index(chunks)
    print(f"Index built with {len(index)} unique terms.")

    # 3. Save to disk
    os.makedirs(STORE_DIR, exist_ok=True)

    print(f"\nSaving to {STORE_DIR}/...")
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"  chunks.json     — {len(chunks)} tables ({os.path.getsize(CHUNKS_FILE) // 1024} KB)")

    with open(TFIDF_FILE, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False)
    print(f"  tfidf_index.json — {len(index)} terms ({os.path.getsize(TFIDF_FILE) // 1024} KB)")

    print("\n✓ Ingestion complete. You can now run query.py.")
    print("=" * 60)


if __name__ == "__main__":
    build_store()
