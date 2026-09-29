import argparse
import os
import sys
import time
from typing import Any

# Ensure app directory is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from eval_dataset import GOLDEN_EVAL_EXAMPLES, sync_golden_dataset
from eval_rag import (
    evaluate_correctness,
    evaluate_faithfulness,
    evaluate_routing_accuracy,
)
from graphs.main_graph import master_app

try:
    from langsmith import Client, evaluate
except ImportError:
    Client = None
    evaluate = None


def run_target(inputs: dict[str, Any]) -> dict[str, Any]:
    """Target function invoked by evaluation runner against the master LangGraph app."""
    question = inputs["question"]
    initial_state = {
        "question": question,
        "retry_count": 0,
        "max_retries": 2
    }
    config = {"configurable": {"thread_id": f"eval-{int(time.time() * 1000)}"}}
    result = master_app.invoke(initial_state, config=config)
    return {
        "final_answer": result.get("final_answer", ""),
        "documents": result.get("documents", []),
        "datasource": result.get("datasource", "vectorstore"),
        "review_status": result.get("review_status", "approved"),
        "retry_count": result.get("retry_count", 0)
    }


def run_local_evaluation_suite():
    """Runs a complete local evaluation benchmark when LangSmith API key is not configured."""
    print("\n" + "=" * 70)
    print("🚀 RUNNING LOCAL AGENTIC RAG EVALUATION BENCHMARK")
    print("=" * 70 + "\n")

    results_summary = []
    faithfulness_scores = []
    correctness_scores = []
    routing_scores = []

    for idx, example in enumerate(GOLDEN_EVAL_EXAMPLES, 1):
        question = example["inputs"]["question"]
        ref_outputs = example["outputs"]
        print(f"[{idx}/{len(GOLDEN_EVAL_EXAMPLES)}] Evaluating: '{question[:55]}...'")

        start_time = time.time()
        actual_output = run_target(example["inputs"])
        latency = round(time.time() - start_time, 2)

        # Run evaluators
        faith_res = evaluate_faithfulness(example["inputs"], actual_output, ref_outputs)
        corr_res = evaluate_correctness(example["inputs"], actual_output, ref_outputs)
        route_res = evaluate_routing_accuracy(example["inputs"], actual_output, ref_outputs)

        faithfulness_scores.append(faith_res["score"])
        correctness_scores.append(corr_res["score"])
        routing_scores.append(route_res["score"])

        results_summary.append({
            "idx": idx,
            "question": question[:35] + "...",
            "datasource": actual_output.get("datasource"),
            "faithfulness": faith_res["score"],
            "correctness": corr_res["score"],
            "routing": route_res["score"],
            "latency": f"{latency}s"
        })

    # Print summary table
    print("\n" + "-" * 75)
    print(f"{'#':<3} | {'Question':<38} | {'Route':<11} | {'Faith':<5} | {'Corr':<5} | {'Lat':<6}")
    print("-" * 75)
    for r in results_summary:
        print(f"{r['idx']:<3} | {r['question']:<38} | {r['datasource']:<11} | {r['faithfulness']:<5.2f} | {r['correctness']:<5.2f} | {r['latency']:<6}")
    print("-" * 75)

    avg_faith = sum(faithfulness_scores) / max(len(faithfulness_scores), 1)
    avg_corr = sum(correctness_scores) / max(len(correctness_scores), 1)
    avg_route = sum(routing_scores) / max(len(routing_scores), 1)

    print("\n📊 OVERALL BENCHMARK METRICS:")
    print(f"  • Average Faithfulness (Anti-Hallucination): {avg_faith * 100:.1f}%")
    print(f"  • Average Correctness (Ground Truth Match):   {avg_corr * 100:.1f}%")
    print(f"  • Routing Classification Accuracy:           {avg_route * 100:.1f}%")
    print("=" * 70 + "\n")


def run_langsmith_cloud_evaluation(dataset_name: str, experiment_prefix: str):
    """Executes the evaluation experiment on LangSmith Cloud."""
    client = Client()
    print(f"Connecting to LangSmith project: {os.getenv('LANGCHAIN_PROJECT', 'agentic-rag-service')}...")

    dataset = sync_golden_dataset(client, dataset_name=dataset_name)

    print(f"Triggering LangSmith experiment: '{experiment_prefix}'...")
    results = evaluate(
        run_target,
        data=dataset_name,
        evaluators=[
            evaluate_faithfulness,
            evaluate_correctness,
            evaluate_routing_accuracy,
        ],
        experiment_prefix=experiment_prefix,
        metadata={
            "service_version": "1.0.0",
            "model": "llama3"
        }
    )

    print("\n✅ LANGSMITH EVALUATION COMPLETE!")
    print("View live telemetry and trace heatmaps on LangSmith: https://smith.langchain.com\n")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agentic RAG LangSmith Evaluation Runner")
    parser.add_argument("--dataset", default="Agentic-RAG-Benchmark-v1", help="Name of the LangSmith dataset")
    parser.add_argument("--prefix", default="crag-reflection-eval", help="Experiment name prefix")
    parser.add_argument("--local", action="store_true", help="Force local evaluation run without cloud sync")
    args = parser.parse_args()

    has_langsmith_key = bool(os.getenv("LANGCHAIN_API_KEY"))

    if args.local or not has_langsmith_key or Client is None or evaluate is None:
        if not has_langsmith_key:
            print("Note: LANGCHAIN_API_KEY is not set. Executing local evaluation benchmark harness.")
        run_local_evaluation_suite()
    else:
        run_langsmith_cloud_evaluation(dataset_name=args.dataset, experiment_prefix=args.prefix)
