from langchain_core.prompts import ChatPromptTemplate

GRADE_DOCUMENTS_SYSTEM_PROMPT = """You are a strict retrieval relevance grader.
Your task is to assess whether the retrieved context passages contain facts relevant to answering the user's question.
If the documents contain relevant keywords or semantic information that can answer or help answer the question, grade as 'yes'.
If the documents are completely off-topic, empty, or fail to address the core inquiry, grade as 'no'.

Respond with a binary assessment ('yes' or 'no')."""

grade_prompt_template = ChatPromptTemplate.from_messages([
    ("system", GRADE_DOCUMENTS_SYSTEM_PROMPT),
    ("human", "Retrieved Documents:\n{context}\n\nUser Question:\n{question}\n\nRelevant (yes/no):")
])

REWRITE_QUERY_SYSTEM_PROMPT = """You are an expert search engine query optimizer.
The user asked a question, but initial internal document retrieval returned insufficient or irrelevant results.
Your job is to rewrite the question into an effective, concise, keyword-rich search query designed for search engines like Google or Bing.
Strip all conversational pleasantries ('tell me', 'can you please', 'I am wondering').
Output ONLY the rewritten search query string."""

rewrite_prompt_template = ChatPromptTemplate.from_messages([
    ("system", REWRITE_QUERY_SYSTEM_PROMPT),
    ("human", "Original Question:\n{question}\n\nOptimized Web Search Query:")
])
