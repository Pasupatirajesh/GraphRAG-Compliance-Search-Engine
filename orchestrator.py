from typing import Literal

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

def route_evaluation(state: RAGState) -> Literal[END, "rewriter"]:
    if state["evaluation_passed"]:
        return END
    # If evaluation fails (hallucination or low relevance), cycle back for query correction
    print("⚠️ Evaluation Failed RAG Triad Thresholds. Rewriting and trying again...")
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
