from typing import Any, Dict, List
from langchain_core.runnables import RunnableConfig
from rank_bm25 import BM25Okapi
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from state import RAGState
from retriever import ProductionHybridRetriever

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
    # 2. Iterate through the LLM expanded queries from phase 4
    for q in state["rewritten_queries"][:2]:
        # Search the production instance passed from main.py
        combined_results.extend(live_db.search(q, top_k=2))
        
    # 3. Deduplicate elements by structural data chunk id
    seen = set()
    deduped = [doc for doc in combined_results if doc["id"] not in seen and not seen.add(doc["id"])]
    
    return {"retrieved_documents": deduped}


def reranker_node(state: RAGState) -> Dict[str, Any]:
    docs = state["retrieved_documents"]
    # Simulated Cross-Encoder/Cohere Rerank logic
    # In production: cohere_client.rerank(query=state["cleaned_query"], documents=docs, top_n=2)
    # Here, we keep top items containing matching tokens to mimic cross-attention scoring
    reranked = sorted(docs, key=lambda d: len(set(d["text"].split()) & set(state["cleaned_query"].split())), reverse=True)
    return {"reranked_documents": reranked[:2]}
