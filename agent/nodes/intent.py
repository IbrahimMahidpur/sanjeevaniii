"""Intent classification node for the medical chatbot.

* Detects the language of the user query using fastText (if the model is available).
* Calls the vLLM endpoint to classify the query into one of the supported intents:
  ``symptom_check``, ``drug_info``, ``lab_report``, ``general`` or ``emergency``.
* Stores ``state['user_language']`` (ISO‑639‑1 code) and ``state['intent']``.

The implementation uses a lightweight keyword fallback for language detection when the FastText model
is missing, and a simple LLM prompt for intent classification. The function is async so it can be
used directly in a LangGraph ``StateGraph``.
"""

import os
import json
from pathlib import Path
from typing import Any, Dict

import httpx

try:
    import fasttext
except ImportError:
    fasttext = None

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FASTTEXT_MODEL_PATH = Path("fasttext/lid.176.bin")
_fasttext_model = None

def _load_fasttext_model():
    global _fasttext_model
    if _fasttext_model is None:
        if fasttext is not None and _FASTTEXT_MODEL_PATH.exists():
            _fasttext_model = fasttext.load_model(str(_FASTTEXT_MODEL_PATH))
        else:
            _fasttext_model = None
    return _fasttext_model

def _detect_language(text: str) -> str:
    """Return ISO‑639‑1 language code (e.g. ``en``, ``hi``)."""
    model = _load_fasttext_model()
    if model:
        # fastText returns labels like "__label__en"
        label = model.predict(text.replace("\n", " "), k=1)[0][0]
        return label.split("__")[-1]
    # Fallback – if any non‑ASCII characters assume non‑English, otherwise English
    return "en" if all(ord(ch) < 128 for ch in text) else "non_en"

# ---------------------------------------------------------------------------
# Intent classification via vLLM
# ---------------------------------------------------------------------------

VLLM_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")

_INTENT_PROMPT_TEMPLATE = """You are a medical assistant. Classify the following user query into one of the categories exactly as one of these strings: \n\nsymptom_check, drug_info, lab_report, general, emergency.\n\nUser query: \n{query}\n\nAnswer with only the category name."""

async def _classify_intent_vllm(query: str) -> str:
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": _INTENT_PROMPT_TEMPLATE.format(query=query)}],
        "max_tokens": 5,
        "temperature": 0.0,
        "stream": False,
    }
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(f"{VLLM_URL}/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip().lower()

# ---------------------------------------------------------------------------
# Node implementation
# ---------------------------------------------------------------------------

async def intent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    query = state.get("query", "")
    # Language detection
    state["user_language"] = _detect_language(query)
    # Intent classification
    try:
        intent = await _classify_intent_vllm(query)
    except Exception:
        # Simple keyword fallback if the LLM call fails
        lowered = query.lower()
        if any(kw in lowered for kw in ["chest pain", "stroke", "seizure", "shortness of breath", "unconscious", "collapse", "heart attack"]):
            intent = "emergency"
        elif any(kw in lowered for kw in ["pain", "fever", "cough", "symptom", "symptoms"]):
            intent = "symptom_check"
        elif any(kw in lowered for kw in ["drug", "medication", "dosage", "prescription"]):
            intent = "drug_info"
        elif any(kw in lowered for kw in ["lab", "report", "test", "result"]):
            intent = "lab_report"
        else:
            intent = "general"
    state["intent"] = intent
    return state
