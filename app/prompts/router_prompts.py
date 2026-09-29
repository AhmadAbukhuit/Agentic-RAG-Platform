from langchain_core.prompts import ChatPromptTemplate

ROUTER_SYSTEM_PROMPT = """You are an expert query routing assistant.
Analyze the user's question and determine the optimal datasource:
- 'vectorstore': For questions regarding internal company policies, internal documents, proprietary procedures, contracts, architecture, or archived knowledge.
- 'web_search': For questions regarding current news, live web information, real-time events, public stock prices, external APIs, or recent topics after the knowledge cutoff.

Respond with the selected datasource."""

router_prompt_template = ChatPromptTemplate.from_messages([
    ("system", ROUTER_SYSTEM_PROMPT),
    ("human", "{question}")
])