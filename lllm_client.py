import json
import urllib.request
import urllib.error
from config import LLM_ENDPOINT, LLM_BEARER, LLM_MODEL


def call_llm(system_prompt: str, user_prompt: str) -> str:
    """
    Call an OpenAI-compatible endpoint with Bearer token auth.
    Uses only Python stdlib (urllib) — no extra packages needed.
    """
    headers = {
        "Content-Type":  "application/json",
        "Authorization": f"Bearer {LLM_BEARER}"
    }

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 1024
    }

    data = json.dumps(payload).encode("utf-8")
    req  = urllib.request.Request(LLM_ENDPOINT, data=data, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return body["choices"][0]["message"]["content"].strip()

    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"LLM returned HTTP {e.code}: {error_body}")

    except urllib.error.URLError as e:
        raise RuntimeError(f"Could not reach endpoint {LLM_ENDPOINT}: {e.reason}")
