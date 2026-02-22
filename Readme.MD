# Set these once — Windows
set LLM_ENDPOINT=http://your-internal-llm/v1/chat/completions
set LLM_BEARER=your-bearer-token-here
set LLM_MODEL=gpt-4

# Step 1 — index your 300 DDL files (run once)
python ingest.py

# Step 2 — query anytime
python query.py "Show all customers with overdue invoices"
