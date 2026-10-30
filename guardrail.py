from typing import Any, Dict

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from state import RAGState
from preprocessor import clean_text

class SafetyAssessment(BaseModel):
    is_safe: bool = Field(description="True if query is safe, False if it requests malicious or unsafe guidance.")
    reason: str = Field(description="Brief explanation of why the query failed safety validation.")

def guardrail_node(state: RAGState) -> Dict[str, Any]:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0).with_structured_output(SafetyAssessment)
    
    system_prompt = (
        "You are a strict compliance and safety filter for a highly regulated document store.\n"
        "Analyze the user query. Reject any queries seeking illegal shortcuts, malicious health advice, "
        "or structural exploits. Be conservative."
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{query}")
    ])
    
    chain = prompt | llm
    result = chain.invoke({"query": state["original_query"]})
    
    return {
        "safety_passed": result.is_safe,
        "safety_feedback": result.reason,
        "cleaned_query": clean_text(state["original_query"])
    }
