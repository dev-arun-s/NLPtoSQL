import json
import re
import os
from collections import defaultdict
from llm_client import call_llm

STORE_DIR = "schema_store"
CHUNKS_FILE = os.path.join(STORE_DIR, "chunks.json")
TFIDF_FILE  = os.path.join(STORE_DIR, "tfidf_index.json")

_chunks = None
_index  = None


def _load():
    global _chunks, _index
    if _chunks is None:
        with open(CHUNKS_FILE) as f:
            _chunks = json.load(f)
        with open(TFIDF_FILE) as f:
            _index = json.load(f)


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9_]+", text.lower())
    return [t for t in tokens if len(t) > 2]


def retrieve_schema(user_query: str, top_k: int = 12) -> tuple[str, list[str]]:
    _load()
    query_tokens = tokenize(user_query)
    scores = defaultdict(float)

    for term in query_tokens:
        if term in _index:
            for doc_id_str, score in _index[term].items():
                scores[int(doc_id_str)] += score

    query_lower = user_query.lower()
    for i, chunk in enumerate(_chunks):
        if chunk["table_name"] in query_lower:
            scores[i] += 5.0
        for col in chunk["columns"]:
            if col in query_lower:
                scores[i] += 1.0
        for fk in chunk["foreign_keys"]:
            if fk in query_lower:
                scores[i] += 2.0

    ranked   = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    selected = [_chunks[i] for i, _ in ranked]

    table_names    = [c["table_name"] for c in selected]
    schema_context = "\n\n".join(
        f"-- Source: {c['source_file']}\n{c['ddl']}" for c in selected
    )
    return schema_context, table_names


def generate_sql(user_query: str, schema_context: str) -> str:
    system_prompt = """You are a SQL expert. Given a database schema and a natural language request:
- Return ONLY the SQL query, no explanation, no markdown fences
- Use explicit JOINs (not implicit)
- Use table aliases
- Handle NULLs appropriately
- If the schema is insufficient, respond with: -- Cannot determine from available schema"""

    user_prompt = f"""SCHEMA:
{schema_context}

REQUEST: {user_query}

SQL:"""

    return call_llm(system_prompt, user_prompt)


def text_to_sql(user_query: str, top_k: int = 12) -> dict:
    schema_context, matched_tables = retrieve_schema(user_query, top_k=top_k)
    sql = generate_sql(user_query, schema_context)
    return {
        "query":          user_query,
        "matched_tables": matched_tables,
        "sql":            sql
    }


if __name__ == "__main__":
    import sys
    user_query = " ".join(sys.argv[1:]) or "Get total revenue per customer for last quarter"
    result = text_to_sql(user_query)
    print(f"\nMatched tables : {result['matched_tables']}")
    print(f"\nGenerated SQL  :\n{result['sql']}") 	 	
