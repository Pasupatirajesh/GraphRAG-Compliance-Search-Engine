from typing import List, Dict, Any, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

class RAGState(TypedDict):
    # Core Queries & Inputs
    original_query: str
    cleaned_query: str
    rewritten_queries: List[str]
    
    # Context Buffers
    retrieved_documents: List[Dict[str, Any]] # Raw matches from BM25 & Vector
    reranked_documents: List[Dict[str, Any]]  # Post-Cohere Rerank filter
    
    # Generated Outputs & Metadata
    generation: str
    safety_passed: bool
    safety_feedback: str
    
    # Evaluation Matrix
    faithfulness_score: float
    relevance_score: float
    evaluation_passed: bool
