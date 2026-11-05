# logger_utils.py — Append-only audit ledger for RAG transactions
# Writes one JSON Lines entry per graph invocation so every query's audit
# trail (safety gate outcome, triad scores, citations, latency) survives on
# disk. Structured for later replay in a dataframe or the dashboard.

import json
import os
from datetime import datetime

AUDIT_LEDGER_PATH = "rag_execution_audit.jsonl"

def log_rag_transaction(user_query: str, final_state: dict, duration_ms: float) -> None:
    """Append one transaction record to the audit ledger.

    Never raises: a logging failure must not take down a completed pipeline
    run — the ledger is observability, not correctness.
    """
    try:
        entries = final_state.get("reranked_documents") or []
        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "query": user_query,
            "safety_passed": final_state.get("safety_passed"),
            "safety_feedback": final_state.get("safety_feedback"),
            "verdict": final_state.get("verdict"),
            "context_coverage": final_state.get("context_coverage"),
            "rewrite_attempts": final_state.get("rewrite_attempts"),
            "faithfulness_score": final_state.get("faithfulness_score"),
            "relevance_score": final_state.get("relevance_score"),
            # Truncate: the ledger is for auditing retrieval behavior, not
            # for storing full completions (those live in LangSmith traces).
            "generation": (final_state.get("generation") or "")[:500],
            "duration_ms": round(duration_ms, 1),
            "citations": [
                {"source": d.get("source"), "page": d.get("page")}
                for d in entries if isinstance(d, dict)
            ],
        }
        with open(AUDIT_LEDGER_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as exc:
        print(f"⚠️ Audit logger could not write ({exc.__class__.__name__}). Pipeline result unaffected.")