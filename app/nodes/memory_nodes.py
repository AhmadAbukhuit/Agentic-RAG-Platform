import re

from langchain_core.messages import AIMessage, HumanMessage
from state.schema import AgentState
from tools.memory_store import recall_user_memories, save_user_memory


def recall_memory(state: AgentState) -> dict:
    """Recalls long-term user preferences and resolves the active question from messages or input."""
    user_id = state.get("user_id")
    question = state.get("question", "")
    messages = state.get("messages", [])

    # If question is empty, extract text from the latest HumanMessage in short-term history
    if not question and messages:
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage) or (hasattr(msg, "type") and msg.type == "human"):
                question = str(msg.content)
                break

    # Recall long-term memories for this user
    memories: list[str] = []
    if user_id:
        memories = recall_user_memories(user_id, limit=5)
        if memories:
            print(f"---LONG-TERM MEMORY RECALLED for user '{user_id}': {len(memories)} facts---")

    return {
        "question": question,
        "user_memories": memories
    }


def persist_memory(state: AgentState) -> dict:
    """Extracts explicit user preferences for long-term storage and appends the final answer to short-term message history."""
    user_id = state.get("user_id")
    question = state.get("question", "")
    final_answer = state.get("final_answer", "")

    # 1. Long-Term Fact Extraction: Detect explicit preferences (e.g., "remember that...", "my role is...")
    if user_id and question:
        preference_patterns = [
            r"(?:remember that|note that|keep in mind that)\s+(.*)",
            r"(?:my name is|i work at|my department is|my company is)\s+(.*)",
            r"(?:i prefer|always format as|always respond in)\s+(.*)",
        ]
        for pattern in preference_patterns:
            match = re.search(pattern, question, re.IGNORECASE)
            if match:
                extracted_fact = match.group(0).strip()
                print(f"---EXTRACTED DURABLE FACT FOR USER '{user_id}': '{extracted_fact}'---")
                save_user_memory(user_id, extracted_fact)

    # 2. Short-Term Memory Update: Append final answer as AIMessage to thread history
    updates = {}
    if final_answer:
        updates["messages"] = [AIMessage(content=final_answer)]

    return updates
