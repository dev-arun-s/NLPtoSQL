"""
app.py — Web UI for Text-to-SQL
Run: python app.py
Then open: http://localhost:5000
"""

from flask import Flask, request, jsonify, render_template_string
import json
import re
import os
import sys
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

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    selected = [_chunks[i] for i, _ in ranked if scores[i] > 0] or _chunks[:top_k]

    table_names = [c["table_name"] for c in selected]
    schema_context = "\n\n".join(
        f"-- Table: {c['table_name']} (from {c['source_file']})\n{c['ddl']}"
        for c in selected
    )
    return schema_context, table_names


def generate_sql(user_query, schema_context):
    system_prompt = """You are a SQL expert assistant working with an enterprise database.
Given a database schema (DDL) and a natural language request:
- Return ONLY the SQL query — no explanations, no markdown code fences
- Use explicit JOINs (never implicit comma joins)
- Use short table aliases
- Handle NULLs appropriately
- If the schema is insufficient, respond with: -- Cannot determine: <brief reason>"""

    user_prompt = f"""DATABASE SCHEMA:
{schema_context}

USER REQUEST:
{user_query}

SQL:"""
    return call_llm(system_prompt, user_prompt)


HTML = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SQL Query Builder</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg:        #0b0d0f;
    --surface:   #111316;
    --border:    #1e2126;
    --border-hi: #2e333b;
    --accent:    #00e5a0;
    --accent-dim:#00e5a022;
    --accent2:   #5b8eff;
    --text:      #e8eaed;
    --muted:     #6b7280;
    --sql-bg:    #0d1117;
  }

  html, body {
    height: 100%;
    background: var(--bg);
    color: var(--text);
    font-family: 'Syne', sans-serif;
    overflow-x: hidden;
  }

  /* Subtle grid background */
  body::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
      linear-gradient(var(--border) 1px, transparent 1px),
      linear-gradient(90deg, var(--border) 1px, transparent 1px);
    background-size: 48px 48px;
    opacity: 0.4;
    pointer-events: none;
    z-index: 0;
  }

  .wrapper {
    position: relative;
    z-index: 1;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 60px 24px 80px;
  }

  /* Header */
  header {
    text-align: center;
    margin-bottom: 52px;
    animation: fadeDown 0.6s ease both;
  }

  .badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: var(--accent-dim);
    border: 1px solid var(--accent);
    color: var(--accent);
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.12em;
    padding: 5px 12px;
    border-radius: 2px;
    margin-bottom: 20px;
    text-transform: uppercase;
  }

  .badge-dot {
    width: 6px; height: 6px;
    background: var(--accent);
    border-radius: 50%;
    animation: pulse 2s infinite;
  }

  h1 {
    font-size: clamp(2.2rem, 5vw, 3.6rem);
    font-weight: 800;
    letter-spacing: -0.03em;
    line-height: 1.05;
    color: var(--text);
  }

  h1 span {
    color: var(--accent);
  }

  .subtitle {
    margin-top: 12px;
    color: var(--muted);
    font-size: 1rem;
    font-weight: 400;
    letter-spacing: 0.01em;
  }

  /* Main card */
  .card {
    width: 100%;
    max-width: 820px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 4px;
    overflow: hidden;
    animation: fadeUp 0.6s ease 0.1s both;
  }

  .card-header {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 14px 20px;
    border-bottom: 1px solid var(--border);
    background: #0e1013;
  }

  .dot { width: 10px; height: 10px; border-radius: 50%; }
  .dot-r { background: #ff5f57; }
  .dot-y { background: #febc2e; }
  .dot-g { background: #28c840; }

  .card-label {
    margin-left: 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: var(--muted);
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }

  .input-area {
    padding: 24px;
    border-bottom: 1px solid var(--border);
  }

  textarea {
    width: 100%;
    background: transparent;
    border: none;
    outline: none;
    color: var(--text);
    font-family: 'Syne', sans-serif;
    font-size: 1.05rem;
    font-weight: 400;
    line-height: 1.65;
    resize: none;
    min-height: 80px;
  }

  textarea::placeholder { color: var(--muted); }

  .input-footer {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 14px 24px;
    background: #0e1013;
  }

  .hint {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: var(--muted);
    display: flex;
    align-items: center;
    gap: 8px;
  }

  kbd {
    background: var(--border-hi);
    border: 1px solid #3a3f47;
    border-radius: 3px;
    padding: 2px 6px;
    font-size: 10px;
    color: var(--text);
  }

  button {
    display: flex;
    align-items: center;
    gap: 8px;
    background: var(--accent);
    color: #000;
    border: none;
    padding: 10px 22px;
    font-family: 'Syne', sans-serif;
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    cursor: pointer;
    border-radius: 2px;
    transition: opacity 0.15s, transform 0.1s;
  }

  button:hover { opacity: 0.88; }
  button:active { transform: scale(0.97); }
  button:disabled { opacity: 0.4; cursor: not-allowed; transform: none; }

  /* Arrow icon */
  .arrow {
    width: 14px; height: 14px;
    transition: transform 0.2s;
  }
  button:not(:disabled):hover .arrow { transform: translateX(3px); }

  /* Result panel */
  .result-panel {
    display: none;
    flex-direction: column;
    animation: fadeUp 0.35s ease both;
  }

  .result-panel.visible { display: flex; }

  .result-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 14px 20px;
    border-top: 1px solid var(--border);
    background: #0e1013;
  }

  .result-meta {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 3px 8px;
    border-radius: 2px;
  }

  .tag-sql {
    background: var(--accent-dim);
    color: var(--accent);
    border: 1px solid var(--accent);
  }

  .tables-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: var(--muted);
  }

  .tables-label span {
    color: var(--accent2);
  }

  .copy-btn {
    background: transparent;
    border: 1px solid var(--border-hi);
    color: var(--muted);
    font-size: 11px;
    padding: 6px 12px;
    letter-spacing: 0.05em;
  }

  .copy-btn:hover { border-color: var(--accent); color: var(--accent); opacity: 1; }

  pre {
    margin: 0;
    padding: 28px;
    background: var(--sql-bg);
    overflow-x: auto;
    border-top: 1px solid var(--border);
    border-bottom-left-radius: 4px;
    border-bottom-right-radius: 4px;
  }

  code {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.875rem;
    font-weight: 400;
    line-height: 1.8;
    color: #e8eaed;
    white-space: pre-wrap;
    word-break: break-word;
  }

  /* SQL syntax highlight */
  .kw  { color: #ff7b72; font-weight: 500; }   /* keywords */
  .fn  { color: #d2a8ff; }                      /* functions */
  .str { color: #a5d6ff; }                      /* strings/aliases */
  .num { color: #f2cc60; }                      /* numbers */
  .cmt { color: #6b7280; font-style: italic; }  /* comments */

  /* Error state */
  .error-box {
    display: none;
    margin: 0;
    padding: 20px 24px;
    background: #1a0d0d;
    border-top: 1px solid #3b1515;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: #ff7b72;
    line-height: 1.6;
  }
  .error-box.visible { display: block; }

  /* Loading spinner */
  .spinner {
    width: 14px; height: 14px;
    border: 2px solid #00000040;
    border-top-color: #000;
    border-radius: 50%;
    animation: spin 0.6s linear infinite;
    display: none;
  }
  .loading .spinner { display: block; }
  .loading .arrow   { display: none; }

  /* History */
  .history {
    width: 100%;
    max-width: 820px;
    margin-top: 32px;
    animation: fadeUp 0.6s ease 0.2s both;
  }

  .history-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: var(--muted);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 12px;
  }

  .history-item {
    padding: 12px 16px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 2px;
    margin-bottom: 8px;
    cursor: pointer;
    transition: border-color 0.15s;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
  }

  .history-item:hover { border-color: var(--border-hi); }

  .history-q {
    font-size: 0.9rem;
    color: var(--text);
    flex: 1;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .history-time {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: var(--muted);
    flex-shrink: 0;
  }

  @keyframes fadeDown { from { opacity:0; transform:translateY(-16px) } to { opacity:1; transform:none } }
  @keyframes fadeUp   { from { opacity:0; transform:translateY(16px)  } to { opacity:1; transform:none } }
  @keyframes spin     { to { transform: rotate(360deg) } }
  @keyframes pulse    { 0%,100%{opacity:1} 50%{opacity:0.3} }
</style>
</head>
<body>
<div class="wrapper">

  <header>
    <div class="badge"><span class="badge-dot"></span>AI-Powered</div>
    <h1>Natural Language<br>to <span>SQL</span></h1>
    <p class="subtitle">Describe what you need — get production-ready SQL instantly</p>
  </header>

  <div class="card">
    <!-- Title bar -->
    <div class="card-header">
      <span class="dot dot-r"></span>
      <span class="dot dot-y"></span>
      <span class="dot dot-g"></span>
      <span class="card-label">query_builder.sql</span>
    </div>

    <!-- Input -->
    <div class="input-area">
      <textarea id="prompt" rows="3"
        placeholder="e.g. Show all customers with overdue invoices in the last 30 days…"
        autofocus></textarea>
    </div>

    <div class="input-footer">
      <span class="hint"><kbd>Ctrl</kbd><kbd>Enter</kbd> to run</span>
      <button id="run-btn" onclick="runQuery()">
        <span>Generate SQL</span>
        <svg class="arrow" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
          <path d="M5 12h14M12 5l7 7-7 7"/>
        </svg>
        <div class="spinner"></div>
      </button>
    </div>

    <!-- Result -->
    <div class="result-panel" id="result-panel">
      <div class="result-header">
        <div class="result-meta">
          <span class="tag tag-sql">SQL</span>
          <span class="tables-label" id="tables-label"></span>
        </div>
        <button class="copy-btn" onclick="copySQL()">Copy</button>
      </div>
      <pre><code id="sql-output"></code></pre>
    </div>

    <!-- Error -->
    <div class="error-box" id="error-box"></div>
  </div>

  <!-- History -->
  <div class="history" id="history-section" style="display:none">
    <div class="history-title">Recent Queries</div>
    <div id="history-list"></div>
  </div>

</div>

<script>
const history = [];

async function runQuery() {
  const prompt = document.getElementById('prompt').value.trim();
  if (!prompt) return;

  const btn       = document.getElementById('run-btn');
  const resultPanel = document.getElementById('result-panel');
  const errorBox  = document.getElementById('error-box');

  // Loading state
  btn.disabled = true;
  btn.classList.add('loading');
  resultPanel.classList.remove('visible');
  errorBox.classList.remove('visible');

  try {
    const res = await fetch('/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: prompt })
    });

    const data = await res.json();

    if (!res.ok || data.error) {
      throw new Error(data.error || `Server error ${res.status}`);
    }

    // Render SQL with syntax highlighting
    document.getElementById('sql-output').innerHTML = highlight(data.sql);

    // Show matched tables
    const tablesLabel = document.getElementById('tables-label');
    if (data.matched_tables && data.matched_tables.length) {
      tablesLabel.innerHTML = 'Tables: ' + data.matched_tables
        .map(t => `<span>${t}</span>`)
        .join(', ');
    } else {
      tablesLabel.innerHTML = '';
    }

    resultPanel.classList.add('visible');
    addToHistory(prompt, data.sql);

  } catch (err) {
    errorBox.textContent = '⚠ ' + err.message;
    errorBox.classList.add('visible');
  } finally {
    btn.disabled = false;
    btn.classList.remove('loading');
  }
}

function highlight(sql) {
  const keywords = [
    'SELECT','FROM','WHERE','JOIN','LEFT','RIGHT','INNER','OUTER','FULL','CROSS',
    'ON','AND','OR','NOT','IN','EXISTS','BETWEEN','LIKE','IS','NULL','AS',
    'GROUP BY','ORDER BY','HAVING','LIMIT','OFFSET','UNION','ALL','DISTINCT',
    'INSERT','INTO','VALUES','UPDATE','SET','DELETE','CREATE','TABLE','DROP',
    'ALTER','INDEX','WITH','CASE','WHEN','THEN','ELSE','END','COALESCE',
    'COUNT','SUM','AVG','MIN','MAX','TOP','ROWNUM'
  ];

  // Escape HTML first
  let out = sql
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');

  // Comments
  out = out.replace(/(--[^\n]*)/g, '<span class="cmt">$1</span>');

  // Strings
  out = out.replace(/'([^']*)'/g, "<span class='str'>'$1'</span>");

  // Numbers
  out = out.replace(/\b(\d+(\.\d+)?)\b/g, '<span class="num">$1</span>');

  // Keywords (case-insensitive, whole word)
  keywords.forEach(kw => {
    const re = new RegExp(`\\b(${kw})\\b`, 'gi');
    out = out.replace(re, '<span class="kw">$1</span>');
  });

  return out;
}

function copySQL() {
  const raw = document.getElementById('sql-output').textContent;
  navigator.clipboard.writeText(raw).then(() => {
    const btn = document.querySelector('.copy-btn');
    btn.textContent = 'Copied!';
    setTimeout(() => btn.textContent = 'Copy', 1800);
  });
}

function addToHistory(query, sql) {
  const now = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  history.unshift({ query, sql, time: now });

  const section = document.getElementById('history-section');
  const list    = document.getElementById('history-list');
  section.style.display = 'block';

  // Keep last 8
  const recent = history.slice(0, 8);
  list.innerHTML = recent.map((h, i) => `
    <div class="history-item" onclick="loadHistory(${i})">
      <span class="history-q">${escHtml(h.query)}</span>
      <span class="history-time">${h.time}</span>
    </div>
  `).join('');
}

function loadHistory(i) {
  document.getElementById('prompt').value = history[i].query;
  document.getElementById('sql-output').innerHTML = highlight(history[i].sql);
  document.getElementById('result-panel').classList.add('visible');
}

function escHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// Ctrl+Enter to submit
document.addEventListener('keydown', e => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') runQuery();
});
</script>
</body>
</html>'''


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/query", methods=["POST"])
def query():
    try:
        data = request.get_json()
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
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500


if __name__ == "__main__":
    print("=" * 50)
    print("  Text-to-SQL Web UI")
    print("  http://localhost:5000")
    print("=" * 50)
    # Pre-load index on startup so first query is fast
    try:
        _load()
        print(f"  Schema loaded: {len(_chunks)} tables ready.")
    except RuntimeError as e:
        print(f"  WARNING: {e}")
    print("=" * 50)
    app.run(debug=False, port=5000)
