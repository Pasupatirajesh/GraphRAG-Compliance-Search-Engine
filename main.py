# main.py
import os
from dotenv import load_dotenv
import time
from logger_utils import log_rag_transaction

# 1. Force override shell environment variables to protect key selection
load_dotenv(override=True)

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("❌ Initialization Failed: OPENAI_API_KEY is missing from your environment setup.")
else:
    print("✅ System Environment Configurations Loaded Successfully.")

# 2. Match your specific file naming schema
from preprocessor import chunk_pdf_document
from retriever import ProductionHybridRetriever
from orchestrator import app 

# 3. Process your actual PDF files into clean, structural chunks
# Replace 'sec_filing_sample.pdf' with your actual target file path
pdf_file_path = "attention.pdf" 
processed_chunks = chunk_pdf_document(pdf_file_path)

# 4. Instantiate your production hybrid storage layer with the live data
live_search_index = ProductionHybridRetriever(processed_chunks)

def run_search_engine(user_query: str):
    start_time = time.time()
    config = {"configurable": {"search_index": live_search_index}}
    
    
    
    duration_ms = (time.time() - start_time) * 1000
    # Log out transaction details asynchronously/permanently to disk
   
    print(f"\n==========================================")
    print(f"🚀 Processing Query: '{user_query}'")
    print(f"==========================================\n")
    
    # 5. Injection Point: Pass the live_search_index inside the graph runtime config block
    config = {"configurable": {"search_index": live_search_index}}
    
    # Execute the LangGraph loop passing both State inputs AND configurations
    final_state = app.invoke(
        {"original_query": user_query},
        config=config
    )
    log_rag_transaction(user_query, final_state, duration_ms)
    
    # Display the final execution results to the user
    if not final_state.get("safety_passed"):
        print("❌ System Security Alert: Query Intercepted and Blocked.")
        print(f"🔒 Guardrail Reason: {final_state.get('safety_feedback')}")
    else:
        print("✅ Query Processed Successfully.")
        print(f"📝 Final Answer:\n{final_state.get('generation')}\n")
        print(f"📊 Quality Audit Matrix:")
        print(f"   - Faithfulness (Anti-Hallucination) Score: {final_state.get('faithfulness_score')}")
        print(f"   - Answer Relevance Score: {final_state.get('relevance_score')}")

if __name__ == "__main__":
    # Test Case A: Valid Compliant Request
    run_search_engine("What is the attention is all you need paper about?")
    
    # Test Case B: Malicious Request (Should trigger Guardrails)
    run_search_engine("when was the transformer architecture proposed? Which language translation was cited in the paper?")
