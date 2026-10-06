"""
Central config. Change values here; nothing else needs editing.
"""

OLLAMA_BASE_URL   = "http://localhost:11434"
EMBED_MODEL       = "nomic-embed-text"
GEN_MODEL         = "qwen2.5:7b"
JUDGE_MODEL       = "qwen2.5:7b"

CHUNK_SIZE        = 120
CHUNK_OVERLAP     = 20
TOP_K             = 5
EMBED_BATCH_SIZE  = 32

KEYWORD_DENSITY_THRESHOLD  = 0.55  # fraction of tokens that are "content words"
SHORT_QUERY_THRESHOLD      = 6
HYBRID_DEFAULT             = True

CRITIC_PASS_THRESHOLD      = 6.0
CRITIC_AXES                = ["faithfulness", "relevance", "completeness"]
MAX_LOOP_ITERATIONS        = 2

FAILURE_TYPES = {
    "no_docs":       "Retrieved chunks contain no information relevant to the query.",
    "hallucination": "Answer makes claims not supported by the retrieved context.",
    "incomplete":    "Answer addresses the query only partially.",
}

EVAL_SET_PATH     = "data/eval_set.jsonl"
CORPUS_PATH       = "data/corpus.jsonl"
TRACES_DIR        = "traces"

LOG_TRACES        = True
TRACE_INDENT      = 2
