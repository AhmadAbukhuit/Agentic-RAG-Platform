import base64
import os
import re
from typing import Any

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

from core.llm import get_vision_llm
from langchain_core.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from state.ingestion_schema import IngestionState
from tools.nosql_db import insert_into_mongodb
from tools.vectorstore import insert_hybrid_indexes


def extract_text_and_ocr(state: IngestionState) -> dict[str, Any]:
    """Extracts text from PDF, using native text extraction with vision LLM fallback for scans."""
    print("---EXTRACTING TEXT & OCR---")
    file_path = state["file_path"]
    full_text = ""

    if not os.path.exists(file_path):
        return {
            "raw_extracted_text": f"File not found: {file_path}",
            "metadata": {"source": file_path, "pages": 0}
        }

    if fitz is None:
        raise ImportError("PyMuPDF (fitz) is not installed. Please install 'pymupdf' to extract PDFs.")

    with fitz.open(file_path) as doc:
        total_pages = len(doc)
        vision_llm = None  # Lazy init only if needed

        for page_num in range(total_pages):
            page = doc[page_num]
            native_text = page.get_text().strip()

            # If page contains ample native digital text, use it directly (100x faster)
            if len(native_text) > 50:
                full_text += f"\n--- Page {page_num + 1} ---\n" + native_text
            else:
                # Scanned page / image: run multimodal OCR
                if vision_llm is None:
                    vision_llm = get_vision_llm()

                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img_base64 = base64.b64encode(pix.tobytes("jpeg")).decode("utf-8")

                message = HumanMessage(
                    content=[
                        {"type": "text", "text": "Extract all text and tabular data from this document page verbatim. Do not summarize."},
                        {"type": "image_url", "image_url": f"data:image/jpeg;base64,{img_base64}"}
                    ]
                )
                try:
                    response = vision_llm.invoke([message])
                    full_text += f"\n--- Page {page_num + 1} (OCR) ---\n" + response.content
                except Exception as e:
                    print(f"Vision OCR failed on page {page_num + 1}: {e}")
                    full_text += f"\n--- Page {page_num + 1} ---\n[OCR failed: {e}]"

    return {
        "raw_extracted_text": full_text,
        "metadata": {"source": file_path, "pages": total_pages}
    }


def clean_ocr_text(state: IngestionState) -> dict[str, Any]:
    """Sanitizes raw extracted text for chunking and embedding."""
    print("---CLEANING EXTRACTED TEXT---")
    text = state.get("raw_extracted_text", "")

    # Normalize multiple newlines and weird whitespaces
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Strip invalid control characters while preserving punctuation and alphanumeric
    text = re.sub(r'[^\w\s.,;:!?()\-/$%@#&+=<>[\]{}"]', '', text)
    cleaned_text = text.strip()

    return {"cleaned_text": cleaned_text}


def chunk_document(state: IngestionState) -> dict[str, Any]:
    """Splits cleaned text into overlapping semantic chunks."""
    print("---CHUNKING DOCUMENT---")
    text = state.get("cleaned_text") or state.get("raw_extracted_text", "")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_text(text) if text else []

    return {"chunks": chunks}


def insert_into_databases(state: IngestionState) -> dict[str, Any]:
    """Persists raw text in MongoDB and vector chunks in Qdrant."""
    print("---PERSISTING TO DATABASES---")
    chunks = state.get("chunks", [])
    metadata = state.get("metadata", {})
    raw_text = state.get("raw_extracted_text", "")

    # 1. Save unchunked document to MongoDB
    doc_id = insert_into_mongodb(raw_text=raw_text, metadata=metadata)

    # 2. Embed and store chunks in Qdrant with linkage to MongoDB doc_id
    enriched_metadata = [
        {**metadata, "mongo_id": doc_id, "chunk_index": i}
        for i, _ in enumerate(chunks)
    ]
    insert_hybrid_indexes(chunks=chunks, metadata=enriched_metadata)

    return {"db_status": "Success"}