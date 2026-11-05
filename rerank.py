from typing import Any, Dict, List
import os
from langchain_core.runnables import RunnableConfig
from rank_bm25 import BM25Okapi
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

import cohere

from state import RAGState
from retriever import ProductionHybridRetriever

# Rerank 5 Fast tier: low-latency cross-encoder tuned for RAG pipelines.
# Override via env (e.g. COHERE_RERANK_MODEL=rerank-v5.0-pro) for max quality.
RERANK_MODEL = os.getenv("COHERE_RERANK_MODEL", "rerank-v5.0-fast")

# Top candidates fed to the cross-encoder after coarse hybrid retrieval
RERANK_TOP_N = 2

_cohere_client = None

def _get_cohere_client():
    """Lazily build the Cohere client. Returns None when no key is configured,
    so graph compiles and CI runs stay network-clean."""
    global _cohere_client
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        return None
    if _cohere_client is None:
        _cohere_client = cohere.ClientV2(api_key=api_key)
    return _cohere_client

class QueryExpansion(BaseModel):
    variations: List[str] = Field(description="3 alternative search variations of the prompt targeting technical vocabulary.")

def query_rewriter_node(state: RAGState) -> Dict[str, Any]:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2).with_structured_output(QueryExpansion)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Expand this search query for a financial/legal/medical document database. Generate alternative phrasings targeting dense industry terminology."),
        ("human", "{query}")
    ])
    
    result = (prompt | llm).invoke({"query": state["cleaned_query"]})
    return {
        "rewritten_queries": result.variations,
        # Circuit-breaker counter: bump once every time the loop re-enters the rewriter
        "rewrite_attempts": state.get("rewrite_attempts", 0) + 1
    }


def retrieve_node(state: RAGState, config: RunnableConfig) -> Dict[str, Any]:
    # 1. Pull the live search index out of the config dynamically passed from main.py
    configurable = config.get("configurable", {})
    live_db = configurable.get("search_index")
    
    if not live_db:
        raise ValueError("❌ Architectural Error: No active 'search_index' found in LangGraph config runtime context.")
        
    combined_results = []
    # 2. Iterate through the LLM expanded queries from phase 4.
    # Coarse recall: take a WIDE shortlist from the hybrid retriever so the
    # downstream cross-encoder actually has candidates to discriminate between.
    # At paper scale (8 chunks) a wide shortlist is redundant; at SEC-filing
    # scale (thousands of chunks) it is what makes reranking worthwhile.
    for q in state["rewritten_queries"][:2]:
        # Search the production instance passed from main.py
        combined_results.extend(live_db.search(q, top_k=5))
        
    # 3. Deduplicate elements by structural data chunk id
    seen = set()
    deduped = [doc for doc in combined_results if doc["id"] not in seen and not seen.add(doc["id"])]
    
    return {"retrieved_documents": deduped}


def _overlap_rerank(docs: List[Dict[str, Any]], query: str, top_n: int) -> List[Dict[str, Any]]:
    """Offline cross-encoder approximation: keep documents sharing the most
    literal tokens with the query. Deterministic and network-free — used as
    the graceful fallback when Cohere is unavailable, and unit-tested."""
    return sorted(
        docs,
        key=lambda d: len(set(d["text"].split()) & set(query.split())),
        reverse=True,
    )[:top_n]

def reranker_node(state: RAGState) -> Dict[str, Any]:
    docs = state["retrieved_documents"]
    query = state["cleaned_query"]
    if not docs:
        return {"reranked_documents": []}

    client = _get_cohere_client()
    if client:
        try:
            response = client.rerank(
                model=RERANK_MODEL,
                query=query,
                documents=[d["text"] for d in docs],
                top_n=RERANK_TOP_N,
            )
            reranked = [
                {**docs[hit.index], "rerank_score": hit.relevance_score}
                for hit in response.results
            ]
            print(f"🎯 Cohere {RERANK_MODEL}: reranked {len(docs)} candidates -> {len(reranked)} passages")
            return {"reranked_documents": reranked}
        except Exception as exc:
            # Fail open: degraded (lexical) ranking beats a crashed pipeline
            print(f"⚠️ Cohere rerank unavailable ({exc.__class__.__name__}: {exc}). Falling back to token-overlap scoring.")

    return {"reranked_documents": _overlap_rerank(docs, query, RERANK_TOP_N)}
