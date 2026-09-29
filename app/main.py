"""FastAPI Application Entrypoint & LangServe API Gateway.

This module initializes the FastAPI server, configures CORS policies,
and mounts LangGraph agent workflows using LangServe to expose standard REST
endpoints (/invoke, /batch, /stream, /playground) for both the main RAG agent
and the document ingestion pipeline.
"""

from core.config import settings
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from graphs.main_graph import master_app
from graphs.subgraphs.ingestion import ingestion_app
from langserve import add_routes

# Initialize the FastAPI application
app = FastAPI(
    title=settings.SERVICE_NAME,
    version=settings.SERVICE_VERSION,
    description=settings.SERVICE_DESCRIPTION,
)

# Configure CORS safely
origins = settings.allowed_origins
allow_all = "*" in origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=not allow_all,  # Browsers forbid allow_credentials=True with wildcard origins
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Expose LangGraph workflows via LangServe
# Main Q&A Agent Endpoint
add_routes(
    app,
    master_app,
    path="/agents",
)

# Ingestion Pipeline Endpoint
add_routes(
    app,
    ingestion_app,
    path="/ingestion",
)


@app.get("/health")
def health_check():
    """Health check probe endpoint for Kubernetes / Docker container monitoring."""
    return {
        "service_name": settings.SERVICE_NAME,
        "service_version": settings.SERVICE_VERSION,
        "status": "healthy",
    }