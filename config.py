import os

# ── Configure these to match your environment ──────────────────────────────────

# Folder containing your DDL .sql files (scanned recursively)
DDL_FOLDER = r"C:\DDL"

# Where the index files will be saved (created automatically)
STORE_DIR = "schema_store"

# Your LLM endpoint
LLM_ENDPOINT = os.getenv("LLM_ENDPOINT", "http://your-internal-llm/v1/chat/completions")

# Bearer token for auth
LLM_BEARER = os.getenv("LLM_BEARER", "your-bearer-token-here")

# Model name your endpoint expects
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4")

# How many top tables to pass to the LLM as context
TOP_K = 12
