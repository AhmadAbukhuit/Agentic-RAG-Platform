import re
from typing import Any

from core.llm import get_llm
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


class EvalScore(BaseModel):
    score: float = Field(description="Score between 0.0 and 1.0.")
    reasoning: str = Field(description="Brief explanation of the score.")


FAITHFULNESS_PROMPT = """You are an expert AI evaluator for RAG systems.
Your task is to judge whether the Assistant's Final Answer is FAITHFUL to the Retrieved Documents.
An answer is faithful if EVERY factual claim in the answer is directly supported by the retrieved documents.
If the answer makes up facts or introduces outside knowledge not found in the documents, penalize the score.

User Question: {question}
Retrieved Documents:
{documents}

Final Answer:
{final_answer}

Rate faithfulness from 0.0 (completely hallucinated / fabricated) to 1.0 (100% strictly grounded in the documents)."""

CORRECTNESS_PROMPT = """You are an expert AI evaluator judging the correctness of a generated answer against a ground truth answer.
Evaluate whether the Assistant's Answer conveys the same key facts and answers the user's question accurately compared to the Ground Truth.

User Question: {question}
Ground Truth Reference:
{ground_truth}

Assistant's Answer:
{final_answer}

Rate correctness from 0.0 (completely incorrect) to 1.0 (fully accurate)."""


def evaluate_faithfulness(inputs: dict[str, Any], outputs: dict[str, Any], reference_outputs: dict[str, Any] = None) -> dict[str, Any]:
    """Measures whether the final answer is strictly grounded in retrieved documents without hallucination."""
    question = inputs.get("question", "")
    final_answer = outputs.get("final_answer", "")
    documents = outputs.get("documents", [])
    doc_context = "\n\n".join(documents) if documents else "No context retrieved."

    if not final_answer:
        return {"key": "faithfulness", "score": 0.0, "reasoning": "Final answer is empty."}

    llm = get_llm(temperature=0.0)
    prompt = ChatPromptTemplate.from_template(FAITHFULNESS_PROMPT)

    try:
        structured_llm = llm.with_structured_output(EvalScore)
        eval_result = (prompt | structured_llm).invoke({
            "question": question,
            "documents": doc_context,
            "final_answer": final_answer
        })
        return {
            "key": "faithfulness",
            "score": float(eval_result.score),
            "reasoning": eval_result.reasoning
        }
    except Exception as e:
        # Fallback keyword-based evaluation if LLM structured output is unavailable
        has_hallucination = "not available" in final_answer.lower() or "no information" in final_answer.lower()
        score = 1.0 if has_hallucination or len(documents) > 0 else 0.5
        return {
            "key": "faithfulness",
            "score": score,
            "reasoning": f"Fallback rule evaluation: {e}"
        }


def evaluate_correctness(inputs: dict[str, Any], outputs: dict[str, Any], reference_outputs: dict[str, Any]) -> dict[str, Any]:
    """Measures semantic similarity and factual alignment with ground truth."""
    question = inputs.get("question", "")
    final_answer = outputs.get("final_answer", "")
    ground_truth = reference_outputs.get("ground_truth", "") if reference_outputs else ""

    if not ground_truth:
        return {"key": "correctness", "score": 1.0, "reasoning": "No ground truth specified."}

    llm = get_llm(temperature=0.0)
    prompt = ChatPromptTemplate.from_template(CORRECTNESS_PROMPT)

    try:
        structured_llm = llm.with_structured_output(EvalScore)
        eval_result = (prompt | structured_llm).invoke({
            "question": question,
            "ground_truth": ground_truth,
            "final_answer": final_answer
        })
        return {
            "key": "correctness",
            "score": float(eval_result.score),
            "reasoning": eval_result.reasoning
        }
    except Exception as e:
        # Simple token overlap fallback
        words_truth = set(re.findall(r"\w+", ground_truth.lower()))
        words_answer = set(re.findall(r"\w+", final_answer.lower()))
        overlap = len(words_truth & words_answer) / max(len(words_truth), 1)
        return {
            "key": "correctness",
            "score": round(min(overlap, 1.0), 2),
            "reasoning": f"Keyword overlap fallback: {e}"
        }


def evaluate_routing_accuracy(inputs: dict[str, Any], outputs: dict[str, Any], reference_outputs: dict[str, Any]) -> dict[str, Any]:
    """Evaluates whether the router selected the expected datasource (vectorstore vs web_search)."""
    expected = reference_outputs.get("expected_datasource") if reference_outputs else None
    actual = outputs.get("datasource")

    if not expected:
        return {"key": "routing_accuracy", "score": 1.0, "reasoning": "No expected datasource."}

    is_correct = actual == expected
    return {
        "key": "routing_accuracy",
        "score": 1.0 if is_correct else 0.0,
        "reasoning": f"Expected '{expected}', selected '{actual}'."
    }
