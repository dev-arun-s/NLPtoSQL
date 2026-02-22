import os

LLM_ENDPOINT = os.getenv("LLM_ENDPOINT", "http://your-internal-llm/v1/chat/completions")
LLM_BEARER   = os.getenv("LLM_BEARER",   "your-bearer-token-here")
LLM_MODEL    = os.getenv("LLM_MODEL",    "gpt-4")  # whatever model name your endpoint expects
