# CriticRAG

Multi-agent RAG pipeline built from scratch. No LangChain, no LlamaIndex.

A critic-driven validation loop wraps standard retrieve → generate, with MoE retrieval routing, self-query reformulation, tool calling, and a pluggable eval harness.

```
Query
  │
  ▼
┌─────────────────────────────────────┐
│         Router Agent (MoE gate)     │  dense / sparse / hybrid / wikipedia / calculate
└────────────────┬────────────────────┘
                 │ top-k chunks
                 ▼
┌─────────────────────────────────────┐
│         Generator Agent             │  self-consistency N=3 (majority vote)
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│         Critic Agent                │  LLM-as-judge: faithfulness / relevance / completeness
└────────────────┬────────────────────┘
                 │
        ┌────────┴──────────┐
        │ pass (score ≥ θ)  │ fail
        ▼                   ▼
     Return             Classify failure
     answer             no_docs → search_wikipedia
                        hallucination → dense retry
                        incomplete → reformulate + retry
```

## Results

Evaluated on HotpotQA distractor split, 100 hard multi-hop questions.

| Pipeline | Exact Match | Token F1 | Latency |
|---|---|---|---|
| Naive RAG | 60.0% | 0.638 | 2.2s/q |
| CriticRAG | 62.0% | 0.663 | 5.6s/q |
| Delta | **+2.0%** | **+0.025** | +3.4s |

Improvement progression:

| Stage | EM | What changed |
|---|---|---|
| Baseline naive | 52% | single-pass RAG |
| + critic loop | 57% | validation loop + reformulation |
| + answer normalization | 62% | dates, abbreviations, suffix matching |
| + self-consistency (N=3) | 62% | majority vote in generator |
| + tool calling | 62%+ | live Wikipedia on corpus misses |

## Stack

- Python 3.11+
- `numpy`, `requests` — only pipeline dependencies
- `pytest`, `datasets` — eval
- Ollama — embeddings (`nomic-embed-text`) + generation + judge (`qwen2.5:7b`)

## Setup

```bash
ollama pull nomic-embed-text
ollama pull qwen2.5:7b
ollama serve

pip install numpy requests pytest datasets

python3 scripts/build_corpus.py --full
python3 -m criticrag "What causes aurora borealis?"
```

## Eval

```bash
# custom corpus eval
python3 scripts/run_eval.py

# HotpotQA distractor benchmark
python3 scripts/build_hotpotqa_distractor.py --n 100 --level hard

# tool calling demo
python3 scripts/demo_tools.py
```

## Structure

```
criticrag/
├── criticrag/
│   ├── agents/
│   │   ├── router.py       MoE gate + tool calling
│   │   ├── generator.py    self-consistency generation
│   │   ├── critic.py       LLM-as-judge + failure taxonomy
│   │   └── tools.py        Wikipedia search + calculator
│   ├── retrieval/
│   │   ├── dense.py        Ollama embedding index
│   │   ├── sparse.py       BM25 from scratch
│   │   └── hybrid.py       reciprocal rank fusion
│   ├── eval/
│   │   └── harness.py      pluggable eval runner
│   ├── utils/
│   │   ├── ollama.py       Ollama API wrapper
│   │   ├── tracer.py       structured execution traces
│   │   ├── corpus.py       corpus loader
│   │   └── config.py       thresholds + model config
│   └── pipeline.py         main ReAct loop
├── scripts/
│   ├── build_corpus.py             Wikipedia corpus builder
│   ├── build_hotpotqa_distractor.py  HotpotQA benchmark
│   ├── run_eval.py                 custom eval harness
│   └── demo_tools.py               tool calling demo
├── tests/                          17 unit tests (mocked, no Ollama needed)
└── data/
    └── eval_set.jsonl              10 seed eval queries
```

## Design

- **No frameworks** — explicit Python throughout; every decision is visible and interviewable
- **ReAct loop** — reasoning (critic) and acting (router) are decoupled across two agents rather than a single chain-of-thought model
- **MoE routing** — heuristic gate over dense/sparse/hybrid/wikipedia/calculate; swappable for a learned gate
- **LLM-as-judge** — handles paraphrase and partial credit; tradeoff is latency and non-determinism, both logged in traces
- **Best-of-N retention** — pipeline keeps the highest-scoring answer across iterations, preventing critic over-rejection from degrading a correct partial answer
- **Tool escalation** — on retry with high proper-noun density, router escalates to live Wikipedia rather than re-querying a corpus that already failed

## License

MIT
