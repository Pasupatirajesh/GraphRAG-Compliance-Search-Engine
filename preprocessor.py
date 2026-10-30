import os
import re
from pypdf import PdfReader
from typing import List, Dict, Any

def clean_text(text: str) -> str:
    
    text = re.sub(r'(Page\s+\d+\s+of\s+\d+|Confidential|Draft|All Rights Reserved)', '', text, flags=re.IGNORECASE)
    # Normalize excessive line breaks, tabs, and duplicate spacing artifacts
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def chunk_pdf_document(pdf_path, chunk_size: int= 1000, chunk_overlap: int = 200) -> List[Dict[str, Any]]:
    """
    Parses a local PDF document into clean, structurally sound overlapping text chunks.
    Tracks structural source metadata for auditing and citation.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Missing resource file at path: {pdf_path}")
        
    reader = PdfReader(pdf_path)
    file_name = os.path.basename(pdf_path)
    
    full_text_buffer = []
    page_mappings = []
    
    # Extract and map characters page-by-page
    for page_idx, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        cleaned_page_text = clean_text(page_text)
        if cleaned_page_text:
            full_text_buffer.append(cleaned_page_text)
            # Track which chunk spans come from which physical document page
            page_mappings.append((page_idx + 1, cleaned_page_text))
            
    # Combine clean text into a sliding context window
    combined_corpus = " ".join(full_text_buffer)
    words = combined_corpus.split(" ")
    
    chunks = []
    chunk_id = 1
    
    # Process sliding window split strings safely
    step = chunk_size - chunk_overlap
    for i in range(0, len(words), step):
        chunk_words = words[i : i + chunk_size]
        if not chunk_words:
            break
            
        chunk_text = " ".join(chunk_words)
        
        # Simple heuristic to tie chunk back to its primary source page
        approx_page = 1
        for page_num, text_content in page_mappings:
            if chunk_words[0] in text_content:
                approx_page = page_num
                break
                
        chunks.append({
            "id": chunk_id,
            "source": file_name,
            "page": approx_page,
            "text": chunk_text
        })
        chunk_id += 1
    print(f"📦 Successfully processed '{file_name}': Extracted {len(chunks)} text fragments.")
    return chunks