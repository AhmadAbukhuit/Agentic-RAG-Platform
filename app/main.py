"""FastAPI Application Entrypoint & LangServe API Gateway.

This module initializes the FastAPI server, configures CORS policies,
and mounts LangGraph agent workflows using LangServe to expose standard REST
endpoints (/invoke, /batch, /stream, /playground) for both the main RAG agent
and the document ingestion pipeline.
"""

import os
import shutil

from core.config import settings
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from graphs.main_graph import master_app
from graphs.subgraphs.ingestion import ingestion_app
from langserve import add_routes
from pydantic import BaseModel, Field


# Clean Pydantic Input Schemas tailored for the LangServe Playground UI
class AgentInput(BaseModel):
    """Clean input schema for the Agent Playground interface."""
    question: str = Field(
        ...,
        description="The question or prompt to ask the Agentic RAG service.",
        examples=["What is the company policy on annual leave?"]
    )
    user_id: str = Field(
        default="",
        description="Optional user ID for recalling long-term user memories and preferences."
    )


class IngestionInput(BaseModel):
    """Clean input schema for the Ingestion Pipeline interface."""
    file_path: str = Field(
        default="sample.txt",
        description="Path or filename to the document to extract, clean, chunk, and index (e.g. 'sample.txt' or 'data/manual.pdf'). Files in the mounted './data' folder are automatically resolved.",
        examples=["sample.txt", "data/handbook.pdf"]
    )
    raw_extracted_text: str = Field(
        default="",
        description="Optional: paste raw text directly into this field to index without needing a local file.",
        examples=[""]
    )


# Configure LangSmith telemetry based on provided credentials and EU/US endpoint
api_key = settings.effective_langsmith_key.strip()
if api_key and not api_key.startswith(("your_", "<")):
    tracing_enabled = "true" if settings.is_langsmith_tracing_enabled else "false"
    endpoint = settings.effective_langsmith_endpoint
    project = settings.effective_langsmith_project

    os.environ["LANGSMITH_TRACING"] = tracing_enabled
    os.environ["LANGCHAIN_TRACING_V2"] = tracing_enabled
    os.environ["LANGSMITH_API_KEY"] = api_key
    os.environ["LANGCHAIN_API_KEY"] = api_key
    os.environ["LANGSMITH_ENDPOINT"] = endpoint
    os.environ["LANGCHAIN_ENDPOINT"] = endpoint
    os.environ["LANGSMITH_PROJECT"] = project
    os.environ["LANGCHAIN_PROJECT"] = project
else:
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"

# Initialize the FastAPI application
app = FastAPI(
    title=settings.SERVICE_NAME,
    version=settings.SERVICE_VERSION,
    description=settings.SERVICE_DESCRIPTION,
)

# Configure CORS safely
origins = settings.cors_origins
allow_all = "*" in origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=not allow_all,  # Browsers forbid allow_credentials=True with wildcard origins
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

def per_req_config_modifier(config: dict, req) -> dict:
    """Ensures a default thread_id exists in configurable for the LangGraph checkpointer."""
    config = dict(config) if config else {}
    configurable = dict(config.get("configurable", {}))
    if "thread_id" not in configurable or not configurable["thread_id"]:
        configurable["thread_id"] = "default-session"
    config["configurable"] = configurable
    return config


# Expose LangGraph workflows via LangServe with explicit clean input types
# Main Q&A Agent Endpoint
add_routes(
    app,
    master_app,
    path="/agents",
    input_type=AgentInput,
    per_req_config_modifier=per_req_config_modifier,
)

# Ingestion Pipeline Endpoint
add_routes(
    app,
    ingestion_app,
    path="/ingestion",
    input_type=IngestionInput,
)


@app.post(
    "/upload",
    tags=["Ingestion"],
    summary="Upload Document (PDF, TXT, MD, etc.)",
    description=(
        "Upload a document file directly from your local machine via browser or HTTP POST. "
        "The file is saved to the shared data directory and automatically processed through "
        "the full ingestion pipeline (extraction, OCR, cleaning, chunking, MongoDB storage, and Qdrant hybrid indexing)."
    ),
)
@app.post(
    "/ingestion/upload",
    tags=["Ingestion"],
    summary="Upload Document (Alias)",
    include_in_schema=False,
)
def upload_document(
    file: UploadFile = File(..., description="Select document to upload (PDF, TXT, MD, CSV, JSON)"),
):
    """Saves the uploaded file to the shared data directory and runs the ingestion workflow."""
    upload_dir = "/app/data" if os.path.exists("/app/data") else "data"
    os.makedirs(upload_dir, exist_ok=True)

    target_path = os.path.join(upload_dir, file.filename)
    try:
        with open(target_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {e}") from e
    finally:
        file.file.close()

    try:
        result = ingestion_app.invoke({"file_path": target_path})
        status = result.get("db_status", "Success")
        chunks = result.get("chunks", [])
        metadata = result.get("metadata", {})
        return {
            "status": status,
            "filename": file.filename,
            "stored_path": target_path,
            "pages": metadata.get("pages", 1),
            "chunks_indexed": len(chunks),
            "metadata": metadata,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion pipeline failure: {e}") from e


@app.get("/health")
def health_check():
    """Health check probe endpoint for Kubernetes / Docker container monitoring."""
    return {
        "service_name": settings.SERVICE_NAME,
        "service_version": settings.SERVICE_VERSION,
        "status": "healthy",
    }