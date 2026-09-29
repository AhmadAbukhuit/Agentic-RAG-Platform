import os
import sys

# Ensure the app directory is in Python path for test discovery
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from core.cache import cache_manager
from eval_dataset import sync_golden_dataset
from eval_rag import (
    evaluate_correctness,
    evaluate_routing_accuracy,
)
from graphs.main_graph import master_app, route_cache, should_revise
from graphs.subgraphs.ingestion import ingestion_app
from graphs.subgraphs.researcher import decide_crag_action, researcher_app
from langchain_core.messages import AIMessage, HumanMessage
from nodes.cache_nodes import check_cache, save_cache
from nodes.crag_nodes import grade_documents
from nodes.ingest_nodes import chunk_document, clean_ocr_text
from nodes.memory_nodes import persist_memory, recall_memory
from nodes.reranker import rerank_documents
from state.ingestion_schema import IngestionState
from state.schema import AgentState
from tools.memory_store import (
    clear_user_memories,
    recall_user_memories,
    save_user_memory,
)
from tools.reranker import rerank_passages


def test_clean_ocr_text():
    raw_ocr = "Hello   world!!!\n\n\n\nThis is a test document with bad artifacts: \x00\x01\x02."
    state: IngestionState = {
        "file_path": "dummy.pdf",
        "raw_extracted_text": raw_ocr,
        "cleaned_text": "",
        "metadata": {},
        "chunks": [],
        "db_status": ""
    }
    result = clean_ocr_text(state)
    cleaned = result["cleaned_text"]
    assert "Hello   world!!!" in cleaned
    assert "\x00" not in cleaned
    assert "\n\n\n\n" not in cleaned


def test_chunk_document():
    sample_text = "This is a sentence. " * 60  # ~1200 characters
    state: IngestionState = {
        "file_path": "dummy.pdf",
        "raw_extracted_text": "",
        "cleaned_text": sample_text,
        "metadata": {},
        "chunks": [],
        "db_status": ""
    }
    result = chunk_document(state)
    chunks = result["chunks"]
    assert len(chunks) >= 2
    assert all(len(c) <= 1000 for c in chunks)


def test_reflection_loop_routing():
    """Verifies that should_revise correctly routes back to drafter on critique and terminates on approval or retry limit."""
    state_revise: AgentState = {
        "review_status": "revision_needed",
        "retry_count": 1,
        "max_retries": 2
    }
    assert should_revise(state_revise) == "drafter"

    state_approved: AgentState = {
        "review_status": "approved",
        "retry_count": 1,
        "max_retries": 2
    }
    assert should_revise(state_approved) == "finalize"

    state_max_retries: AgentState = {
        "review_status": "revision_needed",
        "retry_count": 2,
        "max_retries": 2
    }
    assert should_revise(state_max_retries) == "finalize"


def test_redis_cache_routing():
    """Verifies that route_cache short-circuits to memory persistence when a cached response is found."""
    hit_state: AgentState = {"cache_hit": True}
    assert route_cache(hit_state) == "cache_hit"

    miss_state: AgentState = {"cache_hit": False}
    assert route_cache(miss_state) == "cache_miss"


def test_cache_manager_lifecycle():
    """Tests setting, retrieving, and clearing entries in the CacheManager."""
    test_q = "What is the policy on leave?"
    payload = {"final_answer": "Employees receive 20 days annual leave.", "datasource": "vectorstore"}

    cache_manager.clear()
    assert cache_manager.get(test_q) is None

    # Test cache insertion & node execution
    save_cache({"question": test_q, "final_answer": payload["final_answer"], "datasource": "vectorstore"})
    cached_node_result = check_cache({"question": test_q})

    assert cached_node_result["cache_hit"] is True
    assert cached_node_result["final_answer"] == payload["final_answer"]

    # Test clearing
    cache_manager.clear()
    assert check_cache({"question": test_q})["cache_hit"] is False


def test_long_term_memory_store():
    """Tests saving and recalling durable user preferences across sessions."""
    user = "test_user_42"
    clear_user_memories(user)

    save_user_memory(user, "Prefers responses in bullet points.")
    save_user_memory(user, "Works in Engineering department.")

    memories = recall_user_memories(user)
    assert len(memories) == 2
    assert "Prefers responses in bullet points." in memories

    clear_user_memories(user)
    assert len(recall_user_memories(user)) == 0


def test_memory_nodes():
    """Tests recall_memory and persist_memory node functions."""
    user = "alice_dev"
    clear_user_memories(user)

    # 1. Test persist_memory extracting a preference and appending AIMessage
    state_to_persist: AgentState = {
        "user_id": user,
        "question": "Remember that I prefer python examples.",
        "final_answer": "Understood, all future examples will be in Python."
    }
    persist_updates = persist_memory(state_to_persist)
    assert "messages" in persist_updates
    assert isinstance(persist_updates["messages"][0], AIMessage)

    # 2. Test recall_memory retrieving the extracted preference
    state_to_recall: AgentState = {
        "user_id": user,
        "messages": [HumanMessage(content="How do I connect to Qdrant?")]
    }
    recalled = recall_memory(state_to_recall)
    assert recalled["question"] == "How do I connect to Qdrant?"
    assert any("prefer python examples" in m.lower() for m in recalled["user_memories"])

    clear_user_memories(user)


def test_flashrank_reranker():
    """Tests cross-encoder re-ranking utility and node execution."""
    query = "What is Python?"
    candidate_passages = [
        "Unrelated recipe for baking chocolate cake with flour and sugar.",
        "Python is a high-level programming language known for readability.",
        "Random weather forecast in Seattle on Tuesday morning.",
        "Python syntax allows developers to express concepts in fewer lines.",
        "Sports news about baseball game scores yesterday.",
        "Another random text with no relevance."
    ]

    reranked = rerank_passages(query=query, passages=candidate_passages, top_n=3)
    assert len(reranked) <= 3

    node_result = rerank_documents({
        "question": query,
        "documents": candidate_passages
    })
    assert "documents" in node_result
    assert len(node_result["documents"]) == 4


def test_crag_grading_and_routing():
    """Tests Corrective RAG grading edge and decision routing."""
    # Relevant documents route directly to reranker
    state_relevant: AgentState = {"retrieval_grade": "relevant"}
    assert decide_crag_action(state_relevant) == "rerank_documents"

    # Irrelevant documents trigger query rewriting and web search
    state_irrelevant: AgentState = {"retrieval_grade": "irrelevant"}
    assert decide_crag_action(state_irrelevant) == "rewrite_query"

    # Empty documents in grade_documents node returns irrelevant grade
    empty_grade_result = grade_documents({"question": "Any question", "documents": []})
    assert empty_grade_result["retrieval_grade"] == "irrelevant"


def test_langsmith_evaluators_and_dataset():
    """Tests LangSmith evaluators and dataset generator."""
    dataset = sync_golden_dataset(client=None)
    assert len(dataset) >= 4

    inputs = {"question": "What is PTO policy?"}
    outputs = {
        "final_answer": "Full-time employees receive 20 days annual leave.",
        "documents": ["Employees receive 20 days annual leave."],
        "datasource": "vectorstore"
    }
    ref = {
        "ground_truth": "Full-time employees receive 20 days annual leave.",
        "expected_datasource": "vectorstore"
    }

    # Test routing evaluator
    route_eval = evaluate_routing_accuracy(inputs, outputs, ref)
    assert route_eval["score"] == 1.0

    # Test correctness evaluator
    corr_eval = evaluate_correctness(inputs, outputs, ref)
    assert corr_eval["score"] > 0.5


def test_graph_compilation():
    """Validates that all LangGraph state graphs compile without schema or edge errors."""
    assert master_app is not None
    assert researcher_app is not None
    assert ingestion_app is not None


def test_agent_state_schema():
    state: AgentState = {
        "messages": [HumanMessage(content="Hello"), AIMessage(content="Hi")],
        "user_id": "user_123",
        "user_memories": ["Prefers concise answers"],
        "question": "What is the policy on remote work?",
        "cache_hit": False,
        "datasource": "vectorstore",
        "retrieval_grade": "relevant",
        "rewritten_query": None,
        "documents": ["Policy paragraph 1", "Policy paragraph 2"],
        "draft_answer": "Employees may work remotely 2 days a week.",
        "review_status": "approved",
        "review_feedback": None,
        "retry_count": 0,
        "max_retries": 2,
        "final_answer": "Employees may work remotely 2 days a week."
    }
    assert state["datasource"] == "vectorstore"
    assert state["user_id"] == "user_123"
    assert state["retrieval_grade"] == "relevant"
    assert len(state["messages"]) == 2
    assert len(state["user_memories"]) == 1
