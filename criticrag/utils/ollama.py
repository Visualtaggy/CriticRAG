"""
Thin wrapper around the Ollama HTTP API.
All network calls go through here — makes mocking easy in tests.
"""
from __future__ import annotations

import requests
import numpy as np

from .config import OLLAMA_BASE_URL, EMBED_MODEL, GEN_MODEL, JUDGE_MODEL

def _post(endpoint: str, payload: dict) -> dict:
    url = f"{OLLAMA_BASE_URL}{endpoint}"
    try:
        r = requests.post(url, json=payload, timeout=120)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Cannot reach Ollama. Is `ollama serve` running? "
            f"Expected at {OLLAMA_BASE_URL}"
        )

def embed(texts: list[str], model: str = EMBED_MODEL) -> np.ndarray:
    """
    Returns (N, D) float32 array of raw (un-normalised) embeddings.
    nomic-embed-text expects task prefixes; callers add them.
    """
    data = _post("/api/embed", {"model": model, "input": texts})
    return np.array(data["embeddings"], dtype=np.float32)

def generate(prompt: str, model: str = GEN_MODEL, system: str | None = None,
             temperature: float = 0.0) -> str:
    payload: dict = {"model": model, "prompt": prompt, "stream": False,
                     "options": {"temperature": temperature}}
    if system:
        payload["system"] = system
    data = _post("/api/generate", payload)
    return data["response"].strip()

def chat(messages: list[dict], model: str = GEN_MODEL, temperature: float = 0.0) -> str:
    data = _post("/api/chat", {"model": model, "messages": messages, "stream": False,
                               "options": {"temperature": temperature}})
    return data["message"]["content"].strip()

def judge(prompt: str) -> str:
    return generate(prompt, model=JUDGE_MODEL)
