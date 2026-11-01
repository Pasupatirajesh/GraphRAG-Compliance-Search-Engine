from typing import Any, Dict, List, Literal

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
    # Diagnosis: why the answer missed, so the router can retry vs. stop
    context_coverage: Literal["covered", "missing"] = Field(
        description="Is the information required to answer the question actually present anywhere in the provided context?"
    )
    verdict: Literal["pass", "retry", "unanswerable"] = Field(
        description="pass=both scores at or above 0.75. retry=the answer is wrong or evasive, but the context DOES contain the needed information (a phrasing/generation problem the pipeline can fix). "
                    "unanswerable=the context simply lacks the required information (the question is out of scope for this corpus; an honest 'I do not know' answer is CORRECT, not a failure)."
    )

def evaluation_node(state: RAGState) -> Dict[str, Any]:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0).with_structured_output(TriadEvaluation)
    context = "\n".join([d["text"] for d in state["reranked_documents"]])

    eval_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an independent quality auditor. Judge the generated answer against the source context and original query on three things:\n"
         "1. faithfulness: is the answer strictly grounded in the context (no hallucination)?\n"
         "2. relevance: does it directly address the user's question?\n"
         "3. context_coverage: does the context actually contain the information needed to answer?\n"
         "Crucial distinction: if the context simply lacks the answer, the honest 'I do not know' reply is CORRECT "
         "audited behavior — score it faithful=1.0 and mark the verdict 'unanswerable', NOT a quality failure. "
         "Only mark 'retry' when the context had the information but the answer was wrong, evasive, or off-topic anyway."),
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
        "context_coverage": scores.context_coverage,
        "verdict": scores.verdict,
        "evaluation_passed": passed
    }
