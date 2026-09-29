from langchain_core.prompts import ChatPromptTemplate

DRAFT_SYSTEM_PROMPT = """You are a precise, factual AI assistant specializing in synthesizing information from retrieved context.
Answer the user's question directly and concisely based ONLY on the provided context.
Adhere to the user's personal preferences if provided, while remaining strictly grounded in the context.
If the context does not contain enough information, state clearly that it is not available.
Do not fabricate facts or hallucinate."""

draft_prompt_template = ChatPromptTemplate.from_messages([
    ("system", DRAFT_SYSTEM_PROMPT),
    ("human", """User Preferences & Memories:
{user_memories}

Recent Conversation History:
{chat_history}

Retrieved Context:
{context}

Question:
{question}

Draft Answer:""")
])

REVISE_SYSTEM_PROMPT = """You are a meticulous AI assistant revising a previous response based on critique.
Address the reviewer's feedback carefully. Ensure all claims are strictly grounded in the provided context and respect user preferences.
Do not hallucinate or introduce facts not found in the context."""

revise_prompt_template = ChatPromptTemplate.from_messages([
    ("system", REVISE_SYSTEM_PROMPT),
    ("human", """User Preferences & Memories:
{user_memories}

Retrieved Context:
{context}

Question:
{question}

Previous Draft:
{previous_draft}

Reviewer Feedback:
{feedback}

Revised Answer:""")
])

REVIEW_SYSTEM_PROMPT = """You are a rigorous quality assurance and hallucination reviewer.
Analyze the user's question, the retrieved context, and the draft answer.
Evaluate:
1. Faithfulness: Are there any statements in the draft that CANNOT be verified by the context?
2. Relevance: Does the draft directly answer what was asked?

Decide whether to 'approved' or mark 'revision_needed'. If 'revision_needed', provide concrete critique explaining what must be corrected."""

review_prompt_template = ChatPromptTemplate.from_messages([
    ("system", REVIEW_SYSTEM_PROMPT),
    ("human", "Context:\n{context}\n\nQuestion:\n{question}\n\nDraft Answer:\n{draft_answer}")
])
