from typing import Literal

from core.llm import get_llm
from prompts.generator_prompts import (
    draft_prompt_template,
    review_prompt_template,
    revise_prompt_template,
)
from pydantic import BaseModel, Field
from state.schema import AgentState


class ReviewResult(BaseModel):
    status: Literal["approved", "revision_needed"] = Field(
        default="approved",
        description="Decision: 'approved' if draft is faithful to context and answers question; 'revision_needed' if draft contains hallucinations or omissions."
    )
    feedback: str = Field(
        default="",
        description="Specific critique and instructions for the drafter to correct issues if revision is needed."
    )


def draft_response(state: AgentState) -> dict:
    """LangGraph node for synthesizing or revising an answer incorporating short-term context and long-term memory."""
    question = state.get("question", "")
    docs = state.get("documents", [])
    context = "\n\n".join(docs) if docs else "No context available."
    feedback = state.get("review_feedback")
    previous_draft = state.get("draft_answer", "")

    # Format long-term memories and short-term chat history
    memories_list = state.get("user_memories", [])
    formatted_memories = "\n".join(f"- {m}" for m in memories_list) if memories_list else "None recorded."

    messages = state.get("messages", [])
    # Keep up to the 4 most recent turns (excluding the active prompt if already in question)
    prior_turns = [
        f"{getattr(m, 'type', 'message').capitalize()}: {m.content}"
        for m in messages[-4:]
        if getattr(m, "content", "") != question
    ]
    formatted_history = "\n".join(prior_turns) if prior_turns else "No prior turns in this session."

    llm = get_llm(temperature=0.2)

    # If this is a revision cycle triggered by reviewer critique:
    if feedback and previous_draft:
        print("---REVISING DRAFT BASED ON CRITIQUE & PREFERENCES---")
        chain = revise_prompt_template | llm
        inputs = {
            "user_memories": formatted_memories,
            "context": context,
            "question": question,
            "previous_draft": previous_draft,
            "feedback": feedback,
        }
    else:
        print("---GENERATING INITIAL DRAFT WITH MEMORY & CONTEXT---")
        chain = draft_prompt_template | llm
        inputs = {
            "user_memories": formatted_memories,
            "chat_history": formatted_history,
            "context": context,
            "question": question,
        }

    try:
        response = chain.invoke(inputs)
        draft_content = response.content
    except Exception as e:
        print(f"Warning: Draft LLM generation failed ({e}). Returning fallback draft.")
        draft_content = f"Draft answer for '{question}' (Generated using available context)."

    return {
        "draft_answer": draft_content,
        "review_feedback": None  # Reset feedback once addressed
    }


def review_response(state: AgentState) -> dict:
    """LangGraph node for verifying faithfulness, grading hallucinations, and deciding on revision."""
    print("---EVALUATING DRAFT & DETECTING HALLUCINATIONS---")
    question = state.get("question", "")
    docs = state.get("documents", [])
    draft_answer = state.get("draft_answer", "")
    context = "\n\n".join(docs) if docs else "No context available."
    retry_count = state.get("retry_count", 0) + 1
    max_retries = state.get("max_retries", 2)

    llm = get_llm(temperature=0.0)

    try:
        structured_reviewer = llm.with_structured_output(ReviewResult)
        decision = (review_prompt_template | structured_reviewer).invoke({
            "context": context,
            "question": question,
            "draft_answer": draft_answer
        })
        status = decision.status
        feedback = decision.feedback
    except Exception as e:
        print(f"Structured review failed, evaluating via text heuristic: {e}")
        raw_text = (review_prompt_template | llm).invoke({
            "context": context,
            "question": question,
            "draft_answer": draft_answer
        }).content.lower()
        if "revision" in raw_text or "hallucinat" in raw_text or "incorrect" in raw_text:
            status = "revision_needed"
            feedback = "Draft contains unverified claims or omissions according to context."
        else:
            status = "approved"
            feedback = ""

    # Force approval if max retries exceeded to prevent infinite loops
    if status == "revision_needed" and retry_count >= max_retries:
        print(f"Max revision retries ({max_retries}) reached. Finalizing current draft.")
        status = "approved"
        feedback = None

    print(f"---REVIEW RESULT: status='{status}', retry_count={retry_count}/{max_retries}---")

    result = {
        "review_status": status,
        "review_feedback": feedback if status == "revision_needed" else None,
        "retry_count": retry_count,
    }

    if status == "approved":
        result["final_answer"] = draft_answer

    return result