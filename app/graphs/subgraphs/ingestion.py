"""Document Ingestion Subgraph.

Linear workflow orchestrating document parsing, OCR fallback, text cleaning,
recursive chunking, and dual-store persistence (MongoDB raw documents + Qdrant hybrid vectors).
"""

from langgraph.graph import END, START, StateGraph
from nodes.ingest_nodes import (
    chunk_document,
    clean_ocr_text,
    extract_text_and_ocr,
    insert_into_databases,
)
from state.ingestion_schema import IngestionState

# Initialize the Ingestion Workflow
ingestion_builder = StateGraph(IngestionState)

# Add Nodes
ingestion_builder.add_node("extract", extract_text_and_ocr)
ingestion_builder.add_node("clean", clean_ocr_text)
ingestion_builder.add_node("chunk", chunk_document)
ingestion_builder.add_node("index", insert_into_databases)

# Define Linear Flow: START -> extract -> clean -> chunk -> index -> END
ingestion_builder.add_edge(START, "extract")
ingestion_builder.add_edge("extract", "clean")
ingestion_builder.add_edge("clean", "chunk")
ingestion_builder.add_edge("chunk", "index")
ingestion_builder.add_edge("index", END)

# Compile the Ingestion Subgraph
ingestion_app = ingestion_builder.compile()