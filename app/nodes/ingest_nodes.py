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
    """Extracts text from PDF, TXT, or MD documents, with PyMuPDF and vision LLM OCR fallback for scans."""
    print("---EXTRACTING TEXT & OCR---")
    
    # 1. Allow passing raw text directly to bypass file lookup
    raw_input_text = (state.get("raw_extracted_text") or "").strip()
    if raw_input_text:
        return {
            "raw_extracted_text": raw_input_text,
            "metadata": {"source": state.get("file_path") or "direct_text_input", "pages": 1}
        }

    file_path = (state.get("file_path") or "").strip()
    if not file_path:
        error_msg = "No file_path or raw_extracted_text provided for ingestion."
        return {
            "raw_extracted_text": "",
            "db_status": error_msg,
            "metadata": {"source": "", "pages": 0, "error": error_msg}
        }

    # 2. Smart path resolution (handles Docker container mounts and relative paths)
    candidate_paths = [
        file_path,
        os.path.join("/app/data", os.path.basename(file_path)),
        os.path.join("/app/data", file_path),
        os.path.join("/app", file_path),
        os.path.join("data", os.path.basename(file_path)),
        os.path.join("data", file_path),
    ]
    resolved_path = None
    for p in candidate_paths:
        if p and os.path.exists(p) and os.path.isfile(p):
            resolved_path = os.path.abspath(p)
            break

    if not resolved_path:
        error_msg = (
            f"File not found: '{file_path}'. "
            f"In Docker, place files in the mounted './data' folder (e.g. 'data/{os.path.basename(file_path)}' or '{os.path.basename(file_path)}') "
            f"or upload directly via the API at POST /upload (http://localhost:8000/docs)."
        )
        return {
            "raw_extracted_text": "",
            "db_status": error_msg,
            "metadata": {"source": file_path, "pages": 0, "error": error_msg}
        }

    # 3. Plain text formats (.txt, .md, .markdown, .csv, .json, .log)
    ext = os.path.splitext(resolved_path)[1].lower()
    if ext in [".txt", ".md", ".markdown", ".csv", ".json", ".log", ".yaml", ".yml", ".html"]:
        try:
            with open(resolved_path, "r", encoding="utf-8", errors="replace") as f:
                full_text = f.read()
            return {
                "raw_extracted_text": full_text,
                "metadata": {"source": resolved_path, "pages": 1}
            }
        except Exception as e:
            error_msg = f"Failed to read file: {e}"
            return {
                "raw_extracted_text": "",
                "db_status": error_msg,
                "metadata": {"source": resolved_path, "pages": 0, "error": error_msg}
            }

    # 4. PDF extraction using PyMuPDF (fitz) with Vision OCR fallback
    if fitz is None:
        raise ImportError("PyMuPDF (fitz) is not installed. Please install 'pymupdf' to extract PDFs.")

    full_text = ""
    with fitz.open(resolved_path) as doc:
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
        "metadata": {"source": resolved_path, "pages": total_pages}
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
    existing_status = state.get("db_status")

    if existing_status and "File not found" in existing_status:
        return {"db_status": existing_status}

    if not chunks:
        err = existing_status or "No chunks produced: document text was empty."
        return {"db_status": err}

    # 1. Save unchunked document to MongoDB
    doc_id = insert_into_mongodb(raw_text=raw_text, metadata=metadata)

    # 2. Embed and store chunks in Qdrant with linkage to MongoDB doc_id
    enriched_metadata = [
        {**metadata, "mongo_id": doc_id, "chunk_index": i}
        for i, _ in enumerate(chunks)
    ]
    insert_hybrid_indexes(chunks=chunks, metadata=enriched_metadata)

    return {"db_status": "Success"}