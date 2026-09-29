"""Document Ingestion State Schema.

Defines the IngestionState TypedDict contract used across extraction,
text sanitization, recursive chunking, and dual-database indexing stages.
"""

from typing import Any, TypedDict


class IngestionState(TypedDict, total=False):
    """State contract passed through the document ingestion pipeline.
    
    Attributes:
        file_path: Absolute or relative path to the incoming document (PDF, TXT, etc.).
        raw_extracted_text: Unprocessed text extracted from digital PDF or vision OCR.
        cleaned_text: Sanitized text with OCR artifacts and irregular whitespace stripped.
        metadata: Associated file metadata (source path, page count, upload timestamps).
        chunks: List of overlapping semantic chunks produced by the text splitter.
        db_status: Final status string ('Success' or error description).
    """
    file_path: str
    raw_extracted_text: str
    cleaned_text: str
    metadata: dict[str, Any]
    chunks: list[str]
    db_status: str