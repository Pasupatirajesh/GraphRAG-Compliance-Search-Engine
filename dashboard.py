# dashboard.py
import streamlit as st
import os
from dotenv import load_dotenv

# Ensure environment loads before any asset configurations initialize
load_dotenv(override=True)

from preprocessor import chunk_pdf_document
from retreiver import ProductionHybridRetriever
from orchestrator import app

st.set_page_config(page_title="Self-Correcting GraphRAG Engine", layout="wide")
st.title("🛡️ Enterprise GraphRAG Compliance Search Engine")
st.caption("Powered by LangGraph, ChromaDB, and Real-Time RAG Triad Alignment Verification Loop.")

# Sidebar Configuration for File Loading
st.sidebar.header("📁 Document Ingestion Context Storage")
pdf_file = st.sidebar.text_input("Local PDF Filename Target:", value="attention.pdf")

@st.cache_resource
def initialize_engine(file_path):
    if not os.path.exists(file_path):
        return None
    with st.spinner("Processing PDF Text Structural Slices into Chroma..."):
        chunks = chunk_pdf_document(file_path)
        return ProductionHybridRetriever(chunks)

search_index = initialize_engine(pdf_file)

if not search_index:
    st.sidebar.error(f"⚠️ App cannot find file '{pdf_file}' in project root folder. Please ensure it is uploaded.")
else:
    st.sidebar.success("✅ Database Engine Connected and Primed.")

# Main Application Frame Input Block
user_query = st.text_input("Enter your complex or regulated query below:", placeholder="What are the baseline structural compliance guidelines?")

if st.button("Execute Compliance Search Run") and user_query:
    if not search_index:
        st.error("Please load a valid source document first.")
    else:
        with st.spinner("Executing LangGraph Processing Topology Loops..."):
            # Inject production components smoothly into graph via configuration properties
            config = {"configurable": {"search_index": search_index}}
            final_state = app.invoke({"original_query": user_query}, config=config)
            
            # 1. Evaluate Upstream Safety Gates
            if not final_state.get("safety_passed"):
                st.error("❌ Upstream Compliance Security Intercept Alert!")
                st.warning(f"**Reason for rejection:** {final_state.get('safety_feedback')}")
            else:
                st.success("✅ Security Checks Passed. Synthesis Context Extracted.")
                
                # Split layout into interactive insight columns
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader("📝 Generated Output Validation Synthesis")
                    st.markdown(final_state.get("generation"))
                    
                    st.subheader("📚 Dynamic Citation Chunks Fetched")
                    for doc in final_state.get("reranked_documents", []):
                        with st.expander(f"📖 Source: {doc['source']} | Page Reference: {doc['page']}"):
                            st.write(doc['text'])
                
                with col2:
                    st.subheader("📊 RAG Triad Audit Scores")
                    
                    # Present Metric Dashcards beautifully
                    st.metric(label="Faithfulness (Anti-Hallucination)", value=f"{final_state.get('faithfulness_score') * 100:.1f}%")
                    st.metric(label="Answer Relevance Profile", value=f"{final_state.get('relevance_score') * 100:.1f}%")
                    
                    if final_state.get("evaluation_passed"):
                        st.balloons()
                        st.success("🏆 Verification Verdict: Response fully grounded and safe to display.")
                    else:
                        st.error("⚠️ Verification Verdict: Hallucination or low-relevance threshold hit.")
