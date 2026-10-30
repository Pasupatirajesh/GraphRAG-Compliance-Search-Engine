from typing import Any, Dict, List

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from state import RAGState

def generation_node(state: RAGState) -> Dict[str, Any]:
    context = "\n".join([d["text"] for d in state["reranked_documents"]])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user question based strictly on the provided context. If the context does not contain the answer, state that you do not know. Do not hallucinate."),
        ("human", "Context:\n{context}\n\nQuestion: {query}")
    ])
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
    response = (prompt | llm).invoke({"context": context, "query": state["cleaned_query"]})
    return {"generation": response.content}

class TriadEvaluation(BaseModel):
    faithfulness: float = Field(description="Score from 0.0 to 1.0. Is the response supported entirely by the provided context without extrapolation?")
    relevance: float = Field(description="Score from 0.0 to 1.0. Does the answer directly address the user's specific core question?")

def evaluation_node(state: RAGState) -> Dict[str, Any]:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0).with_structured_output(TriadEvaluation)
    context = "\n".join([d["text"] for d in state["reranked_documents"]])
    
    eval_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an independent quality auditor. Rate the generated answer against the source context and original query."),
        ("human", "Context:\n{context}\nQuery: {query}\nGenerated Answer:\n{generation}")
    ])
    
    scores = (eval_prompt | llm).invoke({
        "context": context, 
        "query": state["cleaned_query"], 
        "generation": state["generation"]
    })
    
    passed = scores.faithfulness >= 0.75 and scores.relevance >= 0.75
    return {
        "faithfulness_score": scores.faithfulness,
        "relevance_score": scores.relevance,
        "evaluation_passed": passed
    }
