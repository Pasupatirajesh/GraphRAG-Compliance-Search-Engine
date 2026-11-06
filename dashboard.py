# dashboard.py
import streamlit as st
import os
from dotenv import load_dotenv

# Ensure environment loads before any asset configurations initialize
load_dotenv(override=True)

from preprocessor import chunk_pdf_document
from retriever import ProductionHybridRetriever
from orchestrator import app

st.set_page_config(page_title="Self-Correcting GraphRAG Engine", layout="wide")

# Tier-2 guard: fail professionally instead of a raw traceback when the user
# forgot their keys (e.g. first `docker run` without --env-file .env).
_required_keys = [
    ("OPENAI_API_KEY", "REQUIRED — powers embeddings, generation, and the evaluation auditor"),
    ("COHERE_API_KEY", "optional — enables the Cohere Rerank 5 stage (falls back to lexical scoring without it)"),
]
_missing = [(k, note) for k, note in _required_keys if not os.getenv(k)]

if any(k == "OPENAI_API_KEY" for k, _ in _missing):
    st.error("⚠️ **API keys required before this app can start.**")
    st.markdown(
        "Copy the bundled `sample.env` to `.env` in the project root and fill in your keys:\n\n"
        "```bash\ncp sample.env .env   # then edit .env\n```\n"
        "When running the Docker image: `docker run -p 8501:8501 --env-file .env graphrag-engine`"
    )
    with st.expander("Key status"):
        st.markdown("\n".join(f"**{k}** — {note}" for k, note in _required_keys))
    st.stop()

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
                # Gate the display on the evaluator's final verdict, not just the
                # safety gate — otherwise a non-compliant generation (LLM answering
                # outside its context, e.g. writing quicksort from thin air) is
                # presented as a green "success" even after the Triad audit fails it.
                verdict = final_state.get("verdict") or "pass"
                rounds = final_state.get("rewrite_attempts", 0)

                if verdict == "pass":
                    st.success("✅ Security Checks Passed. Triad audit passed. Synthesis Context Extracted.")

                    # Split layout into interactive insight columns
                    col1, col2 = st.columns([2, 1])

                    with col1:
                        st.subheader("📝 Generated Output Validation Synthesis")
                        st.markdown(final_state.get("generation"))

                        st.subheader("📚 Dynamic Citation Chunks Fetched")
                        for doc in final_state.get("reranked_documents", []):
                            # Show the cross-encoder score when Cohere rerank served this run
                            rerank_note = f" | 🎯 Rerank score: {doc['rerank_score']:.3f}" if doc.get("rerank_score") is not None else ""
                            with st.expander(f"📖 Source: {doc['source']} | Page Reference: {doc['page']}{rerank_note}"):
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

                elif verdict == "unanswerable":
                    # The corpus genuinely lacks the information. The honest
                    # abstention IS the correct outcome — present it as intended,
                    # not as a failure, and spend no rewrite display noise.
                    st.warning("🟡 Query is out of scope for this corpus. The pipeline answered honestly instead of generating.")
                    st.subheader("📝 Honest Abstention")
                    st.markdown(final_state.get("generation"))
                    st.caption(f"Context coverage: {final_state.get('context_coverage') or 'missing'} · Rewrite rounds: {rounds} · Zero wasted LLM rounds by verdict routing.")

                else:
                    # verdict == "retry" reaching here means the rewrite budget
                    # was exhausted: evaluation kept failing, so the last answer
                    # is withheld as a failed audit, never a green success.
                    st.error("🛑 Quality audit failed after exhausting the rewrite budget. Answer withheld.")
                    st.caption(
                        f"Rewrite rounds: {rounds} · "
                        f"Faithfulness: {final_state.get('faithfulness_score')} · "
                        f"Relevance: {final_state.get('relevance_score')} · "
                        f"Context coverage: {final_state.get('context_coverage') or 'n/a'}"
                    )
                    with st.expander("🔍 Technical detail (last withheld candidate answer)"):
                        st.markdown(final_state.get("generation"))
