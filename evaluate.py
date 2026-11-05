# evaluate.py — Empirical mini-evaluation harness
# Runs a fixed query set through the graph in two modes:
#   self_correction=True  — full corrective loop (production topology)
#   self_correction=False — single-pass, no rewrite cycle
# Emits a markdown table (eval_results.md) comparing verdicts, latency,
# and LLM graph steps per mode, so the Triad loop's value is measurable
# rather than asserted.

import os
import sys
import time
import argparse
from dotenv import load_dotenv

load_dotenv(override=True)

from preprocessor import chunk_pdf_document
from retriever import ProductionHybridRetriever
from orchestrator import app

# Query set: designed to exercise all router exits.
#   in-scope     — should pass on the first pass or be rescued by a rewrite
#   out-of-scope — should terminate fast (unanswerable), ON mode should spend
#                  fewer graph steps than a naive loop would have
#   malformed    — guardrail-adjacent / borderline phrasing
EVAL_QUERY_SET = [
    {"text": "How does scaled dot-product attention work?", "expectation": "in-scope"},
    {"text": "What problem does multi-head attention solve over a single attention head?", "expectation": "in-scope"},
    {"text": "What optimizer and learning rate schedule were used to train the model?", "expectation": "in-scope"},
    {"text": "attention is all you need paper summary", "expectation": "in-scope (terse phrasing)"},
    {"text": "What is today's date?", "expectation": "out-of-scope"},
    {"text": "Can you recommend a good pizza restaurant nearby?", "expectation": "out-of-scope"},
    {"text": "Write me a recursive quicksort in Python.", "expectation": "out-of-scope (code-generation request)"},
]

def load_query_set(path):
    """Load a pip-separated query spec (query|expectation|category) from a
    text file, skipping blanks and # comments. See sec_eval_queries.txt."""

    entries = []
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|")]
        text, expectation = parts[0], parts[1]
        category = parts[2] if len(parts) > 2 else ""
        entries.append({"text": text, "expectation": expectation, "category": category})
    return entries

def run_one(query, index, self_correction):
    config = {
        "configurable": {
            "search_index": index,
            "self_correction_enabled": self_correction,
        }
    }
    start = time.time()
    final_state = app.invoke({"original_query": query}, config=config)
    duration = time.time() - start
    return {
        "duration_s": duration,
        "verdict": final_state.get("verdict") or ("blocked" if not final_state.get("safety_passed") else "n/a"),
        "attempts": final_state.get("rewrite_attempts") or 0,
        "faithfulness": final_state.get("faithfulness_score"),
        "relevance": final_state.get("relevance_score"),
        "answer": (final_state.get("generation") or "")[:80],
    }

def main():
    parser = argparse.ArgumentParser(description="A/B evaluation harness for the GraphRAG engine")
    parser.add_argument("--queries", default=None,
                        help="pip-separated query spec file (query|expectation|category); defaults to the built-in attention-paper set")
    parser.add_argument("--pdf", default="attention.pdf",
                        help="PDF to chunk and index for the run")
    args = parser.parse_args()

    query_set = load_query_set(args.queries) if args.queries else EVAL_QUERY_SET

    print(f"Building evaluation index from {args.pdf}...")
    chunks = chunk_pdf_document(args.pdf)
    index = ProductionHybridRetriever(chunks)

    rows = []
    for spec in query_set:
        for mode in (True, False):
            label = "ON " if mode else "OFF"
            print(f"  ▪ [{'correction ' + label}] {spec['text']}")
            try:
                result = run_one(spec["text"], index, mode)
            except Exception as exc:  # keep the table alive if one cell fails
                result = {"duration_s": 0, "verdict": f"ERROR: {exc}", "attempts": 0,
                          "faithfulness": None, "relevance": None, "answer": ""}
            result.update({"query": spec["text"], "expectation": spec["expectation"], "mode": label})
            rows.append(result)

    # Markdown table: one row per query per mode for direct comparison
    header = (
        "| Query | Expected | Loop | Verdict | Rewrite rounds | Faithfulness | Relevance | Time (s) |\n"
        "|---|---|---|---|---|---|---|---|"
    )
    lines = [header]
    for r in rows:
        lines.append(
            f"| {r['query']} | {r['expectation']} | {r['mode']} | {r['verdict']} "
            f"| {r['attempts']} | {r['faithfulness']} | {r['relevance']} | {r['duration_s']:.1f} |"
        )

    table = "\n".join(lines)
    with open("eval_results.md", "w") as fh:
        fh.write("# Mini Evaluation: Self-Correction ON vs OFF\n\n")
        fh.write(f"Corpus: `{args.pdf}` ({len(chunks)} chunks). ")
        fh.write("Query set: " + (f"`{args.queries}`" if args.queries else "built-in demo set") + ".\n\n")
        fh.write(table + "\n")
    print("\n✅ Results written to eval_results.md:\n")
    print(table)

if __name__ == "__main__":
    main()