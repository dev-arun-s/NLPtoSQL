"""
app.py — Web UI for Text-to-SQL
Run: python app.py
Then open: http://localhost:5000
"""

from flask import Flask, request, jsonify, render_template
import json
import re
import os
from collections import defaultdict
from config import STORE_DIR, TOP_K
from llm_client import call_llm

app = Flask(__name__)

CHUNKS_FILE = os.path.join(STORE_DIR, "chunks.json")
TFIDF_FILE  = os.path.join(STORE_DIR, "tfidf_index.json")

_chunks = None
_index  = None


def _load():
    global _chunks, _index
    if _chunks is not None:
        return
    if not os.path.exists(CHUNKS_FILE) or not os.path.exists(TFIDF_FILE):
        raise RuntimeError("Index not found. Please run ingest.py first.")
    with open(CHUNKS_FILE, encoding="utf-8") as f:
        _chunks = json.load(f)
    with open(TFIDF_FILE, encoding="utf-8") as f:
        _index = json.load(f)


def tokenize(text):
    tokens = re.findall(r'[a-z0-9_]+', text.lower())
    return [t for t in tokens if len(t) >= 3]


def retrieve_schema(user_query, top_k=TOP_K):
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
            if len(col) > 3 and col in query_lower:
                scores[i] += 1.0
        for fk in chunk["foreign_keys"]:
            if fk in query_lower:
                scores[i] += 2.0

    ranked   = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    selected = [_chunks[i] for i, _ in ranked if scores[i] > 0] or _chunks[:top_k]

    table_names    = [c["table_name"] for c in selected]
    schema_context = "\n\n".join(
        "-- Table: " + c["table_name"] + " (from " + c["source_file"] + ")\n" + c["ddl"]
        for c in selected
    )
    return schema_context, table_names


def generate_sql(user_query, schema_context):
    system_prompt = (
        "You are a SQL expert assistant working with an enterprise database.\n"
        "Given a database schema (DDL) and a natural language request:\n"
        "- Return ONLY the SQL query, no explanations, no markdown code fences\n"
        "- Use explicit JOINs (never implicit comma joins)\n"
        "- Use short table aliases\n"
        "- Handle NULLs appropriately\n"
        "- If the schema is insufficient, respond with: -- Cannot determine: <brief reason>"
    )
    user_prompt = "DATABASE SCHEMA:\n" + schema_context + "\n\nUSER REQUEST:\n" + user_query + "\n\nSQL:"
    return call_llm(system_prompt, user_prompt)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/query", methods=["POST"])
def query():
    try:
        data       = request.get_json()
        user_query = (data or {}).get("query", "").strip()

        if not user_query:
            return jsonify({"error": "Query cannot be empty."}), 400

        schema_context, matched_tables = retrieve_schema(user_query)
        sql = generate_sql(user_query, schema_context)

        return jsonify({
            "sql":            sql,
            "matched_tables": matched_tables
        })

    except RuntimeError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        return jsonify({"error": "Unexpected error: " + str(e)}), 500


if __name__ == "__main__":
    print("=" * 50)
    print("  Text-to-SQL Web UI")
    print("  http://localhost:5000")
    print("=" * 50)
    try:
        _load()
        print("  Schema loaded: " + str(len(_chunks)) + " tables ready.")
    except RuntimeError as e:
        print("  WARNING: " + str(e))
    print("=" * 50)
    app.run(debug=False, port=5000)
