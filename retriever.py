# retriever.py
import os
import hashlib
import chromadb
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from langchain_openai import OpenAIEmbeddings

class ProductionHybridRetriever:
    def __init__(self, corpus: List[Dict[str, Any]], db_path: str = "./chroma_db"):
        self.corpus = corpus

        # 1. Initialize Lexical Layer (BM25) for exact keyword matches
        self.tokenized_corpus = [doc["text"].lower().split(" ") for doc in self.corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        # 2. Initialize Persistent Chroma Database client
        self.chroma_client = chromadb.PersistentClient(path=db_path)

        # Scope the collection to THIS corpus via a content fingerprint, so
        # ingesting a different document (e.g. swapping the demo paper for an
        # 80k-word SEC filing) gets a fresh namespace instead of serving stale
        # vectors from whatever was indexed before. Re-ingesting the same
        # document reuses its collection and skips re-embedding entirely.
        corpus_fp = hashlib.md5(
            "|".join(d["source"] + ":" + d["text"] for d in self.corpus).encode()
        ).hexdigest()[:8]
        collection_name = f"regulated_docs_{corpus_fp}"

        # Get or create a collection for your vector assets
        self.collection = self.chroma_client.get_or_create_collection(name=collection_name)

        # 3. Use OpenAI Embeddings to populate the persistent database if empty
        self.embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

        if self.collection.count() == 0 and corpus:
            print(f"💾 Populating persistent Chroma collection '{collection_name}' with document vectors...")
            # Namespace IDs by source document: every PDF's chunker restarts ids
            # at 1, so without the source prefix two filings ingested together
            # would silently deduplicate each other's chunks.
            ids = [f"{doc['source']}:{doc['id']}" for doc in corpus]
            texts = [doc["text"] for doc in corpus]

            # Format metadata dictionaries safely for Chroma filtering mechanics
            metadatas = [{"source": doc["source"], "page": doc["page"]} for doc in corpus]

            # Generate raw embedding lists using your OpenAI key
            embeddings = self.embeddings_model.embed_documents(texts)

            self.collection.add(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas
            )
            print(f"✅ Successfully indexed {self.collection.count()} vectors permanently to disk.")

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        # A. Lexical (BM25) Scoring
        tokenized_query = query.lower().strip().split(" ")
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k]
        lexical_results = [self.corpus[idx] for idx in top_bm25_indices]

        # B. Semantic Vector Scoring via Persistent Chroma Index
        query_vector = self.embeddings_model.embed_query(query)
        vector_results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k
        )

        # Convert Chroma's return payload back into your standard dictionary structure
        semantic_results = []
        if vector_results and vector_results['documents']:
            for idx in range(len(vector_results['documents'][0])):
                semantic_results.append({
                    # Namespaced id like "attention.pdf:3" — unique across sources
                    "id": vector_results['ids'][0][idx],
                    "text": vector_results['documents'][0][idx],
                    "source": vector_results['metadatas'][0][idx]['source'],
                    "page": vector_results['metadatas'][0][idx]['page']
                })

        # C. Combine and deduplicate across both retrieval families
        combined = lexical_results + semantic_results
        seen = set()
        # Lexical corpus["id"] is an int; semantic ids are namespaced strings
        # like "attention.pdf:3". Normalize both to (source, numeric id).
        def _key(doc):
            raw = str(doc["id"])
            num = int(raw.split(":")[-1]) if ":" in raw else int(raw)
            return (doc["source"], num)
        deduped = [doc for doc in combined if _key(doc) not in seen and not seen.add(_key(doc))]

        return deduped[:top_k]
