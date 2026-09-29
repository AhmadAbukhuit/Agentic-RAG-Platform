import os
from typing import Any

GOLDEN_EVAL_EXAMPLES: list[dict[str, Any]] = [
    {
        "inputs": {
            "question": "What is the policy on annual leave and PTO rollover for full-time employees?"
        },
        "outputs": {
            "ground_truth": "Full-time employees receive 20 days of paid annual leave per year. A maximum of 5 unused days can be rolled over into the next calendar year.",
            "expected_datasource": "vectorstore"
        },
        "metadata": {
            "category": "internal_policy",
            "eval_type": "faithfulness"
        }
    },
    {
        "inputs": {
            "question": "What are the latest updates and breaking changes in the most recent version of LangGraph?"
        },
        "outputs": {
            "ground_truth": "Recent LangGraph updates introduce unified checkpointer interfaces, improved streaming APIs (astream_events), and enhanced functional API patterns.",
            "expected_datasource": "web_search"
        },
        "metadata": {
            "category": "live_information",
            "eval_type": "relevance"
        }
    },
    {
        "inputs": {
            "question": "How does the company reimburse personal travel expenses on Mars colonization missions?"
        },
        "outputs": {
            "ground_truth": "The provided context does not contain any information regarding Mars colonization or extraterrestrial travel expense reimbursement.",
            "expected_datasource": "vectorstore"
        },
        "metadata": {
            "category": "hallucination_trap",
            "eval_type": "anti_hallucination"
        }
    },
    {
        "inputs": {
            "question": "How does Qdrant hybrid retrieval combine dense semantic vectors with BM25 sparse vectors?"
        },
        "outputs": {
            "ground_truth": "Qdrant hybrid retrieval combines dense vectors (capturing semantic intent) and sparse vectors (capturing exact keyword matches like BM25) using Reciprocal Rank Fusion (RRF).",
            "expected_datasource": "vectorstore"
        },
        "metadata": {
            "category": "technical_architecture",
            "eval_type": "correctness"
        }
    }
]


def sync_golden_dataset(client=None, dataset_name: str = "Agentic-RAG-Benchmark-v1"):
    """Syncs the golden benchmark dataset to LangSmith if API credentials are valid."""
    api_key = os.getenv("LANGCHAIN_API_KEY", "")
    if not api_key or client is None:
        print("LangSmith API key not configured or offline mode. Returning local benchmark examples.")
        return GOLDEN_EVAL_EXAMPLES

    try:
        # Check if dataset already exists in LangSmith
        if client.has_dataset(dataset_name=dataset_name):
            print(f"Dataset '{dataset_name}' already exists in LangSmith. Fetching existing dataset.")
            return client.read_dataset(dataset_name=dataset_name)

        print(f"Creating golden benchmark dataset '{dataset_name}' in LangSmith...")
        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description="Golden evaluation dataset for Agentic RAG service measuring faithfulness, relevance, and hallucination resistance."
        )

        for example in GOLDEN_EVAL_EXAMPLES:
            client.create_example(
                inputs=example["inputs"],
                outputs=example["outputs"],
                metadata=example["metadata"],
                dataset_id=dataset.id
            )

        print(f"Successfully uploaded {len(GOLDEN_EVAL_EXAMPLES)} golden examples to LangSmith.")
        return dataset
    except Exception as e:
        print(f"Warning: Failed to sync dataset to LangSmith: {e}. Falling back to local examples.")
        return GOLDEN_EVAL_EXAMPLES
