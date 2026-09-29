from typing import Literal

from core.llm import get_llm
from prompts.router_prompts import router_prompt_template
from pydantic import BaseModel, Field
from state.schema import AgentState


class RouteDecision(BaseModel):
    datasource: Literal["vectorstore", "web_search"] = Field(
        default="vectorstore",
        description="Routing destination: 'vectorstore' for internal knowledge or 'web_search' for current/external info."
    )


def route_question(state: AgentState) -> dict:
    """Evaluates the user question and routes to either vectorstore or web_search."""
    question = state.get("question", "")
    print(f"---ROUTING QUESTION: '{question[:50]}...'---")

    llm = get_llm(temperature=0.0)

    try:
        structured_router = llm.with_structured_output(RouteDecision)
        chain = router_prompt_template | structured_router
        decision = chain.invoke({"question": question})
        selected_datasource = decision.datasource
    except Exception as e:
        print(f"Structured output failed, falling back to heuristic/text parsing: {e}")
        # Robust fallback if local model doesn't support tool/structured calls natively
        raw_output = (router_prompt_template | llm).invoke({"question": question}).content.lower()
        if "web_search" in raw_output or "web" in raw_output:
            selected_datasource = "web_search"
        else:
            selected_datasource = "vectorstore"

    print(f"---ROUTED TO: {selected_datasource}---")
    return {"datasource": selected_datasource}