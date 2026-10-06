from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import requests

DEFAULT_TOPICS = [
    "Aurora", "Black_hole", "CRISPR", "Photosynthesis", "Quantum_entanglement",
    "Plate_tectonics", "Vaccine", "Neutron_star", "Dark_matter", "Exoplanet",
    "World_War_II", "French_Revolution", "Roman_Empire", "Cold_War",
    "Silk_Road", "Moon_landing", "Renaissance", "Byzantine_Empire",
    "Large_language_model", "Generative_pre-trained_transformer",
    "Internet", "Semiconductor", "Blockchain", "Robotics",
    "Jazz", "Olympic_Games", "Climate_change", "Urbanization",
    "Chess", "Coffee", "Vaccine_hesitancy",
]

WIKI_API = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
CHUNK_SIZE = 120
CHUNK_OVERLAP = 20

def fetch_summary(title: str) -> str | None:
    url = WIKI_API.format(title=title)
    try:
        r = requests.get(url, timeout=10, headers={"User-Agent": "CriticRAG/1.0"})
        if r.status_code == 200:
            data = r.json()
            return data.get("extract", "")
        print(f"  [warn] {title}: HTTP {r.status_code}", file=sys.stderr)
    except requests.RequestException as e:
        print(f"  [warn] {title}: {e}", file=sys.stderr)
    return None

def fetch_sections(title: str, retries: int = 3) -> str | None:
    """Fetch full article text via the MediaWiki API with retry on 429."""
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,
        "exsectionformat": "plain",
        "exlimit": 1,
        "format": "json",
        "redirects": 1,
    }
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=30,
                             headers={"User-Agent": "CriticRAG/1.0 (research project)"})
            if r.status_code == 200:
                pages = r.json().get("query", {}).get("pages", {})
                for page in pages.values():
                    extract = page.get("extract", "")
                    if extract and len(extract) > 200:
                        return extract
                return None  # 200 but empty — don't retry
            elif r.status_code == 429:
                wait = 5 * (attempt + 1)
                print(f"  [429 rate limit] {title} — waiting {wait}s...", file=sys.stderr)
                time.sleep(wait)
            else:
                print(f"  [warn] {title}: HTTP {r.status_code}", file=sys.stderr)
                return None
        except requests.RequestException as e:
            print(f"  [warn] {title}: {e}", file=sys.stderr)
            time.sleep(3)
    return None

def clean(text: str) -> str:
    """Remove wiki markup artifacts and excess whitespace."""
    text = re.sub(r"={2,}.*?={2,}", " ", text)
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def chunk(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    step = max(size - overlap, 1)
    chunks = []
    for i in range(0, len(words), step):
        c = " ".join(words[i : i + size])
        if len(c.split()) >= 10:
            chunks.append(c)
        if i + size >= len(words):
            break
    return chunks

def build(topics: list[str], out_path: str, use_full: bool = False) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    total_chunks = 0
    skipped = []
    with open(out, "w") as f:
        for title in topics:
            print(f"  fetching: {title}")
            text = ""
            if use_full:
                text = fetch_sections(title) or ""
                if not text:
                    time.sleep(1.0)
                    text = fetch_sections(title) or ""
                if not text:
                    text = fetch_summary(title) or ""
                    if text:
                        print(f"  [fallback→summary] {title}")
            else:
                text = fetch_summary(title) or ""
            text = clean(text)
            if not text:
                print(f"  [skip] {title}: empty", file=sys.stderr)
                skipped.append(title)
                continue
            chunks_written = 0
            for c in chunk(text):
                record = {"source": title.replace("_", " "), "text": c}
                f.write(json.dumps(record) + "\n")
                total_chunks += 1
                chunks_written += 1
            print(f"  → {chunks_written} chunks")
            time.sleep(1.5 if use_full else 0.2)

    print(f"\nWrote {total_chunks} chunks → {out}")
    print(f"Fetched: {len(topics) - len(skipped)}/{len(topics)} articles")
    if skipped:
        print(f"Skipped: {skipped}")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topics", default=None, help="One topic per line text file")
    parser.add_argument("--out", default="data/corpus.jsonl")
    parser.add_argument("--full", action="store_true", help="Fetch full article (slower)")
    args = parser.parse_args()

    if args.topics:
        topics = Path(args.topics).read_text().splitlines()
        topics = [t.strip() for t in topics if t.strip()]
    else:
        topics = DEFAULT_TOPICS

    print(f"Building corpus from {len(topics)} Wikipedia articles...")
    build(topics, args.out, use_full=args.full)

if __name__ == "__main__":
    main()
