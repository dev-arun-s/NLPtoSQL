import os
import json
import re
import math
import sqlparse
from collections import defaultdict

DDL_FOLDER = r"C:\DDL"
STORE_DIR = "schema_store"
CHUNKS_FILE = os.path.join(STORE_DIR, "chunks.json")
TFIDF_FILE = os.path.join(STORE_DIR, "tfidf_index.json")


# ── DDL Parsing ────────────────────────────────────────────────────────────────

def parse_file(filepath: str) -> list[dict]:
    """Parse one DDL file; may contain multiple CREATE TABLE statements."""
    with open(filepath, encoding="utf-8", errors="ignore") as f:
        content = f.read()

    chunks = []
    try:
        statements = sqlparse.split(content)
    except Exception:
        statements = [content]  # fallback: treat whole file as one statement

    for stmt in statements:
        stmt = stmt.strip()
        if not stmt:
            continue

        parsed = sqlparse.parse(stmt)[0]
        if parsed.get_type() != "CREATE":
            continue

        match = re.search(
            r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(?:\w+\.)?(\w+)",
            stmt, re.IGNORECASE
        )
        if not match:
            continue

        table_name = match.group(1).lower()
        columns = extract_columns(stmt)
        fk_refs = re.findall(r"REFERENCES\s+(?:\w+\.)?(\w+)", stmt, re.IGNORECASE)
        comments = re.findall(r"--.*", stmt)

        chunks.append({
            "table_name": table_name,
            "source_file": os.path.basename(filepath),
            "columns": columns,
            "foreign_keys": [f.lower() for f in fk_refs],
            "comments": comments,
            "ddl": stmt,
        })

    return chunks


def extract_columns(stmt: str) -> list[str]:
    """Best-effort column name extraction from DDL."""
    keywords = {
        "PRIMARY", "FOREIGN", "UNIQUE", "INDEX", "KEY",
        "CONSTRAINT", "CHECK", "CREATE", "TABLE"
    }
    cols = []
    for line in stmt.splitlines():
        line = line.strip()
        match = re.match(r"^(\w+)\s+\w+", line)
        if match and match.group(1).upper() not in keywords:
            cols.append(match.group(1).lower())
    return cols


def scan_all_ddl(folder: str) -> list[dict]:
    """Walk folder and parse every .sql file."""
    all_chunks = []
    files = [
        os.path.join(root, f)
        for root, _, files in os.walk(folder)
        for f in files
        if f.lower().endswith(".sql")
    ]
    print(f"Found {len(files)} SQL files in {folder}")

    for filepath in files:
        try:
            chunks = parse_file(filepath)
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"  Warning: could not parse {filepath}: {e}")

    return all_chunks


# ── TF-IDF Index (pure Python) ─────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumeric, remove short tokens."""
    tokens = re.findall(r"[a-z0-9_]+", text.lower())
    return [t for t in tokens if len(t) > 2]


def build_tfidf_index(chunks: list[dict]) -> dict:
    """
    Build a simple TF-IDF index.
    For each term, store {chunk_index: tf_idf_score}.
    """
    N = len(chunks)

    # Build searchable text per chunk
    def chunk_text(c):
        return " ".join([
            c["table_name"] * 3,           # boost table name
            " ".join(c["columns"]) * 2,    # boost columns
            " ".join(c["foreign_keys"]),
            " ".join(c["comments"]),
            c["ddl"]
        ])

    corpus = [tokenize(chunk_text(c)) for c in chunks]

    # Document frequency
    df = defaultdict(int)
    for doc_tokens in corpus:
        for term in set(doc_tokens):
            df[term] += 1

    # TF-IDF per document
    index = defaultdict(dict)  # term → {doc_id: score}
    for doc_id, doc_tokens in enumerate(corpus):
        tf = defaultdict(int)
        for t in doc_tokens:
            tf[t] += 1
        for term, freq in tf.items():
            idf = math.log((N + 1) / (df[term] + 1)) + 1
            index[term][doc_id] = (freq / len(doc_tokens)) * idf

    return dict(index)


# ── Main ───────────────────────────────────────────────────────────────────────

def build_store():
    os.makedirs(STORE_DIR, exist_ok=True)

    print("Scanning DDL files...")
    chunks = scan_all_ddl(DDL_FOLDER)
    print(f"Parsed {len(chunks)} tables total.")

    print("Building TF-IDF index...")
    index = build_tfidf_index(chunks)

    with open(CHUNKS_FILE, "w") as f:
        json.dump(chunks, f, indent=2)

    with open(TFIDF_FILE, "w") as f:
        json.dump(index, f)

    print(f"Done. {len(chunks)} tables indexed.")
    print(f"  → {CHUNKS_FILE}")
    print(f"  → {TFIDF_FILE}")


if __name__ == "__main__":
    build_store()
