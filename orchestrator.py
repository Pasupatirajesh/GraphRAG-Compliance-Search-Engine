from typing import Literal

from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END

from state import RAGState
from guardrail import guardrail_node
from rerank import query_rewriter_node, retrieve_node, reranker_node
from llmjudge import generation_node, evaluation_node
# Initialize Topology Flow Graph
workflow = StateGraph(RAGState)

# Add Processing Nodes
workflow.add_node("guardrail", guardrail_node)
workflow.add_node("rewriter", query_rewriter_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("rerank", reranker_node)
workflow.add_node("generate", generation_node)
workflow.add_node("evaluate", evaluation_node)

# Add Node Operational Mappings
workflow.set_entry_point("guardrail")

# Define Routing Conditions
def route_guardrail(state: RAGState) -> Literal["rewriter", END]:
    if state["safety_passed"]:
        return "rewriter"
    return END

MAX_REWRITE_ATTEMPTS = 2

def route_evaluation(state: RAGState, config: RunnableConfig | None = None) -> Literal[END, "rewriter"]:
    # A/B harness hook: configurable["self_correction_enabled"] (default True) lets
    # evaluate.py run the same query set with the corrective loop disabled.
    cfg = (config or {}).get("configurable", {}) or {}
    if not cfg.get("self_correction_enabled", True):
        return END
    verdict = state.get("verdict")
    if verdict == "pass":
        return END
    if verdict == "unanswerable":
        # Out-of-scope question (e.g. "What is today's date?" against a corpus
        # with no date info): rewriting can never conjure missing context.
        # The honest "I do not know" IS the correct answer — stop immediately.
        print("ℹ️ Query is out of scope for this corpus. No rewrite, honest answer accepted.")
        return END
    if state.get("rewrite_attempts", 0) >= MAX_REWRITE_ATTEMPTS:
        # Backstop for hallucination/phrasing failures the rewriter could not fix
        # (and for any run where the evaluator's 'unanswerable' judgment missed).
        print("🛑 Rewrite budget exhausted. Returning 'cannot answer' result instead of looping.")
        return END
    # If evaluation fails while the context DOES have the info, cycle back for query correction
    print("⚠️ Answer quality below threshold. Rewriting and trying again...")
    return "rewriter"

# Connect Structural Edges
workflow.add_conditional_edges("guardrail", route_guardrail)
workflow.add_edge("rewriter", "retrieve")
workflow.add_edge("retrieve", "rerank")
workflow.add_edge("rerank", "generate")
workflow.add_edge("generate", "evaluate")
workflow.add_conditional_edges("evaluate", route_evaluation)

# Compile Executable Application Graph
app = workflow.compile()
