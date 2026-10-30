# Self-Correcting GraphRAG Compliance Engine

A production-grade, stateful search and retrieval engine engineered specifically to navigate highly regulated datasets (e.g., SEC financial filings, clinical journals, or public case law). 

This system moves away from basic, linear retrieval-augmented generation patterns. It implements an advanced multi-stage pipeline utilizing **LangGraph** for resilient multi-node flow control, **ChromaDB** and **BM25** for hybrid lexical-semantic retrieval, and an autonomous runtime **RAG Triad Verification Loop** to intercept hallucinations before they reach an end-user.

---

## 🏗️ Architectural Topology Map

```text
                [ User Query ]
                      │
                      ▼
               ┌──────────────┐
               │  Guardrail   │◄─── Deterministic Regex Parsing
               └──────┬───────┘     & Safety Check Prompting
                      │
            Is Query Safe to Run?
             ├── No  ──► [ Reject Node ] ──► (Exit Process)
             └── Yes ──► [ Query Rewriter ]
                               │
                               ▼
                        [ Hybrid Search ] ──► BM25 (Exact) + ChromaDB (Dense)
                               │              (text-embedding-3-small)
                               ▼
                        [ Context Rerank ]
                               │
                               ▼
                        [ Generate Answer ]
                               │
                               ▼
                        [ RAG Triad Eval ] ──► Metrics: Faithfulness 
                               │               & Answer Relevance
                     Did Response Pass?
                      ├── Yes ──► (Display Output to User)
                      └── No  ──► [ Fallback Loop ] ──► Cycle back to Rewriter
```

---

## 🛠️ Core Engineering Highlights

### 1. Stateful Multi-File Architecture
The system is built on a modular design pattern that isolates independent computational stages into explicit handlers, matching enterprise development practices:
* `state.py`: Defines the strictly typed `RAGState` payload passed across nodes.
* `preprocessor.py`: Uses deterministic regular expressions to scrub document noise (confidentiality stamps, structural page headers, formatting artifacts) and chunks text accurately.
* `retreiver.py`: Implements a multi-tenant retrieval index blending lexical token mapping via **BM25Okapi** with dense vector indices via a local persistent **ChromaDB** client.
* `guardrail.py`: Acts as an upstream compliance gatekeeper checking prompt logic *before* triggering expensive database lookups.
* `rerank.py`: Handles multi-query expansion variations and scoring density text reductions.
* `llmjudge.py`: Generates the final output response and drives the automated evaluation step.
* `orchestrator.py`: Compiles the stateful multi-node **LangGraph** execution workflow.

### 2. Autonomous RAG Triad Verification & Self-Correction
Rather than trusting LLM outputs blindly, the engine scores completions at runtime against the **RAG Triad (Faithfulness and Answer Relevance)** using an LLM-as-a-Judge pattern. If a completion drops below a strict evaluation threshold, LangGraph dynamically flags the transaction, blocks the output, routes the state back into the query rewriter, and attempts self-corrected context collection.

### 3. Application Visibility, Ingestion, & Monitoring
* **Interactive Dashboard:** Built a complete web UI interface using **Streamlit** to visualize system operations, view specific page-level source citations, and display live evaluation dials.
* **Audit Logger:** Appends real-time metrics, safety statuses, and execution durations to an append-only JSON Lines ledger (`rag_execution_audit.jsonl`).
* **CI/CD Automation:** Configured a **GitHub Actions** automation workflow (`python-tests.yml`) to automatically execute the pipeline unit test suite on every single code commit.

---

## ⚡ Technical Stack

* **Orchestration & Graphs:** LangGraph, LangChain
* **Vector Index Storage:** ChromaDB (Local Disk Persistence)
* **Keyword Index Matching:** Rank-BM25 (BM25Okapi)
* **LLM Core / Embeddings:** OpenAI GPT-4o, GPT-4o-mini, `text-embedding-3-small`
* **Data Parsing:** PyPDF + Python Regex Native Library
* **UI & Testing:** Streamlit, Python Unittest, GitHub Actions CI/CD

---

## 🚀 Quick Start Guide

### 1. Set Up the Isolated Environment
```bash
# Spin up your environment and activate it
python3 -m venv rag_env
source rag_env/bin/activate  # On Windows use: rag_env\Scripts\activate

# Install the exact required package stack
pip install --upgrade pip setuptools wheel
pip install langgraph langchain langchain-openai rank-bm25 pydantic pypdf chromadb python-dotenv streamlit
```

### 2. Configure Local Keys
Create a `.env` configuration file in your project's root folder:
```env
OPENAI_API_KEY=sk-proj-yourActualSecretKeyGoesHere
```

### 3. Ingest and Run the UI Dashboard
Place a target regulatory document snippet (e.g., `sec_filing_sample.pdf`) into your root directory, then execute the interface application:
```bash
streamlit run dashboard.py
```
---

## 📈 Sample Execution Output Trace

```text
✅ System Environment Configurations Loaded Successfully.
📦 Successfully processed 'sec_filing_sample.pdf': Extracted 12 text fragments.
💾 Populating persistent Chroma collection with document vectors...
✅ Successfully indexed 12 vectors permanently to disk.

==========================================
🚀 Processing Query: 'What are Section 404 baseline structural compliance constraints?'
==========================================

✅ Query Processed Successfully.
📝 Final Answer:
Section 404 compliance mandates continuous structural logging of validation and internal reporting protocols...

📊 Quality Audit Matrix:
   - Faithfulness (Anti-Hallucination) Score: 1.0
   - Answer Relevance Score: 1.0
```
