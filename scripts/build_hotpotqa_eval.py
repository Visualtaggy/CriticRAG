from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

def load_hotpotqa(split: str = "validation", streaming: bool = True):
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: datasets not installed. Run: pip install datasets")
        sys.exit(1)

    print(f"Loading HotpotQA ({split}, fullwiki)...")
    return load_dataset("hotpot_qa", "fullwiki", split=split, streaming=streaming)

def is_multihop(sample: dict) -> bool:
    """Requires at least 2 distinct supporting articles."""
    titles = sample.get("supporting_facts", {}).get("title", [])
    return len(set(titles)) >= 2

def is_factoid(sample: dict) -> bool:
    """Skip yes/no answers — they don't stress retrieval meaningfully."""
    return sample["answer"].strip().lower() not in ("yes", "no")

def to_eval_entry(sample: dict, idx: int) -> dict:
    supporting = list(set(sample.get("supporting_facts", {}).get("title", [])))
    return {
        "id": f"hpqa_{idx:03d}",
        "query": sample["question"],
        "expected": sample["answer"],
        "rubric": (
            f"The correct answer is: {sample['answer']}. "
            f"This is a {sample['type']} question requiring information from: "
            f"{', '.join(supporting)}. "
            f"Award points for correct answer and supporting reasoning."
        ),
        "source": "hotpotqa",
        "level": sample["level"],
        "type": sample["type"],
        "supporting_articles": supporting,
    }

def build(
    n: int = 50,
    level: str = "hard",
    out_path: str = "data/eval_set_hotpotqa.jsonl",
    skip_yesno: bool = True,
) -> list[dict]:
    ds = load_hotpotqa()
    entries = []
    scanned = 0

    for sample in ds:
        scanned += 1
        if scanned > 5000:
            break

        if not is_multihop(sample):
            continue
        if level != "all" and sample["level"] != level:
            continue
        if skip_yesno and not is_factoid(sample):
            continue

        entries.append(to_eval_entry(sample, len(entries)))

        if len(entries) >= n:
            break

    print(f"Scanned {scanned} samples, kept {len(entries)} (target: {n})")

    if not entries:
        print("WARNING: no samples matched filters. Try --level all or --no-skip-yesno")
        sys.exit(1)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    print(f"Wrote {len(entries)} eval entries → {out}")
    return entries

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50, help="Number of questions to keep")
    parser.add_argument("--level", default="hard", choices=["hard", "medium", "easy", "all"])
    parser.add_argument("--out", default="data/eval_set_hotpotqa.jsonl")
    parser.add_argument("--replace", action="store_true",
                        help="Also overwrite data/eval_set.jsonl with these entries")
    parser.add_argument("--no-skip-yesno", action="store_true",
                        help="Include yes/no answer questions")
    args = parser.parse_args()

    entries = build(
        n=args.n,
        level=args.level,
        out_path=args.out,
        skip_yesno=not args.no_skip_yesno,
    )

    if args.replace:
        replace_path = "data/eval_set.jsonl"
        with open(replace_path, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")
        print(f"Also replaced {replace_path}")

    print("\nSample questions:")
    for e in entries[:3]:
        print(f"  [{e['id']}] ({e['level']}/{e['type']}) {e['query']}")
        print(f"         answer: {e['expected']}")

if __name__ == "__main__":
    main()
