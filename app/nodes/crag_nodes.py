from typing import Literal

from core.llm import get_llm
from prompts.crag_prompts import grade_prompt_template, rewrite_prompt_template
from pydantic import BaseModel, Field
from state.schema import AgentState


class GradeDecision(BaseModel):
    score: Literal["yes", "no"] = Field(
        default="yes",
        description="Binary decision: 'yes' if documents contain information relevant to the question, 'no' otherwise."
    )
    reasoning: str = Field(
        default="",
        description="Brief justification of the relevance assessment."
    )


def grade_documents(state: AgentState) -> dict:
    """Evaluates whether retrieved documents are relevant to the user query."""
    question = state.get("question", "")
    documents = state.get("documents", [])

    print(f"---CRAG: GRADING RELEVANCE OF {len(documents)} RETRIEVED PASSAGES---")

    if not documents:
        print("---CRAG: No documents retrieved. Marking as irrelevant.---")
        return {"retrieval_grade": "irrelevant"}

    context = "\n\n".join(documents[:4])
    llm = get_llm(temperature=0.0)

    try:
        structured_grader = llm.with_structured_output(GradeDecision)
        decision = (grade_prompt_template | structured_grader).invoke({
            "context": context,
            "question": question
        })
        grade = "relevant" if decision.score.lower() == "yes" else "irrelevant"
        print(f"---CRAG EVALUATION: score='{decision.score}', reasoning='{decision.reasoning}'---")
    except Exception as e:
        print(f"Warning: Structured grading failed ({e}). Using heuristic parsing.")
        raw_output = (grade_prompt_template | llm).invoke({
            "context": context,
            "question": question
        }).content.lower()
        grade = "relevant" if "yes" in raw_output else "irrelevant"

    print(f"---CRAG GRADE: {grade.upper()}---")
    return {"retrieval_grade": grade}


def rewrite_query(state: AgentState) -> dict:
    """Rewrites the question into an optimized web search query when internal retrieval is insufficient."""
    question = state.get("question", "")
    print(f"---CRAG: REWRITING QUERY FOR WEB SEARCH FALLBACK: '{question[:50]}'---")

    llm = get_llm(temperature=0.0)
    chain = rewrite_prompt_template | llm

    try:
        response = chain.invoke({"question": question})
        rewritten = response.content.strip().strip('"').strip("'")
    except Exception as e:
        print(f"Warning: Query rewrite failed ({e}). Using original question.")
        rewritten = question

    print(f"---CRAG REWRITTEN QUERY: '{rewritten}'---")
    return {
        "rewritten_query": rewritten,
        "question": rewritten,
        "documents": []  # Clear stale irrelevant documents for fresh web search
    }
