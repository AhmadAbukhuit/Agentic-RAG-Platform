"""Agent State Schema Definitions.

This module defines the central TypedDict schema (AgentState) used by the
master LangGraph workflow to maintain short-term conversational context,
long-term user preferences, retrieved documents, routing flags, reflection loop
counters, and final synthesized answers.
"""

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    """Central state dictionary passed across all nodes in the master agent workflow.
    
    Attributes:
        messages: Conversation history automatically accumulated via add_messages reducer.
        question: The active user query string.
        user_id: Unique user identifier for cross-session long-term memory recall.
        user_memories: List of recalled durable facts/preferences for this user.
        cache_hit: Flag indicating whether query was served directly from Redis cache.
        datasource: Selected routing target ('vectorstore' or 'web_search').
        retrieval_grade: CRAG quality assessment ('relevant' or 'irrelevant').
        rewritten_query: Search-engine optimized query generated during CRAG fallback.
        documents: List of retrieved and cross-encoder re-ranked context passages.
        draft_answer: Initial synthesized response from the drafter node.
        review_status: Quality status from reviewer ('approved' or 'revision_needed').
        review_feedback: Specific critique instructions from reviewer for self-correction.
        retry_count: Number of reflection revision cycles completed so far.
        max_retries: Upper bound on revision attempts before forced finalization.
        final_answer: Grounded, approved final answer delivered to the client.
    """
    # Short-Term Conversation Memory (Scoped to thread_id)
    messages: Annotated[list[BaseMessage], add_messages]
    question: str

    # Long-Term User Memory (Scoped across sessions to user_id)
    user_id: str
    user_memories: list[str]

    # Caching & Routing
    cache_hit: bool
    datasource: str

    # Corrective RAG (CRAG) Quality & Transformation
    retrieval_grade: str
    rewritten_query: str

    # Context & Synthesis (Re-ranked document passages)
    documents: list[str]
    draft_answer: str

    # Reflection Loop & Review
    review_status: str
    review_feedback: str
    retry_count: int
    max_retries: int

    # Final Output
    final_answer: str