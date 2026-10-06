from __future__ import annotations

import ast
import operator
import re
import time

import requests

WIKI_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"

_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.BitXor: operator.pow,
    ast.USub: operator.neg,
}

def _safe_eval(node):
    if isinstance(node, ast.Constant):
        return node.n
    if isinstance(node, ast.BinOp):
        op = _SAFE_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported operator: {node.op}")
        return op(_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        op = _SAFE_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported operator: {node.op}")
        return op(_safe_eval(node.operand))
    raise ValueError(f"Unsupported node: {node}")

def calculate(expression: str) -> list[tuple[float, str, str]]:
    """
    Safely evaluate a math expression.
    Returns result as a single hit so the pipeline interface is uniform.
    """
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _safe_eval(tree.body)
        return [(1.0, "calculator", f"The result of {expression} is {result}.")]
    except Exception as e:
        return [(0.0, "calculator", f"Could not evaluate '{expression}': {e}")]

def _search_wikipedia_title(query: str) -> str | None:
    """Find the best matching Wikipedia title for a query via opensearch."""
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "opensearch",
        "search": query,
        "limit": 3,
        "format": "json",
    }
    try:
        r = requests.get(url, params=params, timeout=10,
                         headers={"User-Agent": "CriticRAG/1.0"})
        if r.status_code == 200:
            results = r.json()
            titles = results[1] if len(results) > 1 else []
            return titles[0] if titles else None
    except requests.RequestException:
        pass
    return None

def _extract_entity(query: str) -> str:
    """
    Extract the most likely entity name from a query for Wikipedia lookup.
    Picks the longest run of capitalised words (skipping first word).
    Falls back to full query.
    """
    words = query.split()
    if len(words) < 2:
        return query
    best_run = []
    current_run = []
    for w in words[1:]:
        clean = w.strip("?,.'\"")
        if clean and clean[0].isupper():
            current_run.append(clean)
        else:
            if len(current_run) > len(best_run):
                best_run = current_run[:]
            current_run = []
    if len(current_run) > len(best_run):
        best_run = current_run
    return " ".join(best_run) if best_run else query

def search_wikipedia(query: str) -> list[tuple[float, str, str]]:
    """
    Live Wikipedia summary lookup.
    Extracts entity name from query for better title matching.
    Returns up to 3 hits as (score, title, summary_text).
    """
    hits = []
    entity = _extract_entity(query)
    candidates = [entity.replace(" ", "_")]
    found_title = _search_wikipedia_title(entity) or _search_wikipedia_title(query)
    if found_title:
        candidates.insert(0, found_title.replace(" ", "_"))

    seen = set()
    for title in candidates[:2]:
        if title in seen:
            continue
        seen.add(title)
        try:
            url = WIKI_SUMMARY_URL.format(title=title)
            r = requests.get(url, timeout=10,
                             headers={"User-Agent": "CriticRAG/1.0"})
            if r.status_code == 200:
                data = r.json()
                extract = data.get("extract", "").strip()
                display = data.get("title", title)
                if extract and len(extract) > 50:
                    hits.append((0.9, f"wikipedia:{display}", extract[:500]))
            time.sleep(0.1)
        except requests.RequestException:
            pass

    if not hits:
        hits = [(0.0, "wikipedia", "No Wikipedia article found for this query.")]
    return hits
