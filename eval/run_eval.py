"""Ablation runner. Regenerates one row of the ablation table by running the
full retrieve(+generate) pipeline over the 50-question eval set -- nothing in
the ablation.json output is hand-typed.

Usage:
    python -m eval.run_eval --config dense_512
    python -m eval.run_eval --config hybrid_rerank_agent
    python -m eval.run_eval --all
"""
from __future__ import annotations

import argparse
import json
import statistics
import time

from langchain_core.messages import HumanMessage, SystemMessage

from src import config as app_config
from src.agent.graph import ask as agent_ask
from src.agent.nodes import CITATION_RE, GENERATE_SYSTEM_NO_TOOLS
from src.retrieval.pipeline import ABLATION_CONFIGS, retrieve as pipeline_retrieve
from eval.metrics import (
    aggregate_generation_metrics,
    aggregate_retrieval_metrics,
    judge_faithfulness,
    judge_relevance,
    looks_like_refusal,
)

EVAL_SET_PATH = app_config.EVAL_DIR / "eval_set.json"
RESULTS_PATH = app_config.EVAL_RESULTS_DIR / "ablation.json"


def _simple_generate(question: str, hits: list[dict]) -> str:
    """Non-agentic single-shot generation for the pure-retrieval ablation
    rows (no tools, no retry loop -- that's isolated to hybrid_rerank_agent)."""
    llm = app_config.get_chat_llm()
    context = "\n\n".join(f"[{h['chunk_id']}] {h['text']}" for h in hits)
    messages = [
        SystemMessage(content=GENERATE_SYSTEM_NO_TOOLS),
        HumanMessage(content=f"Question: {question}\n\nRetrieved context:\n{context or '(none)'}"),
    ]
    response = llm.invoke(messages)
    return response.content or ""


def run_config(config_name: str, eval_set: list[dict]) -> dict:
    retrieval_rows = []
    judged_rows = []
    latencies = []
    retry_counts = []

    for item in eval_set:
        question = item["question"]
        is_unanswerable = item["type"] == "unanswerable"
        start = time.perf_counter()

        if config_name == "hybrid_rerank_agent":
            result = agent_ask(question)
            hits = [
                {"chunk_id": d.metadata["chunk_id"], "doc_id": d.metadata["doc_id"]}
                for d in result["documents"]
            ]
            answer = result.get("answer", "")
            retry_counts.append(result.get("retry_count", 0))
        else:
            hits = pipeline_retrieve(question, config_name, top_k=5)
            answer = _simple_generate(question, hits)
            retry_counts.append(0)

        elapsed = time.perf_counter() - start
        latencies.append(elapsed)

        retrieved_doc_ids = list(dict.fromkeys(h["doc_id"] for h in hits))
        retrieval_rows.append(
            {"retrieved_doc_ids": retrieved_doc_ids, "relevant_doc_ids": item["relevant_doc_ids"]}
        )

        context_text = "\n\n".join(h.get("text", "") for h in hits)
        judgment = judge_faithfulness(question, context_text, answer)
        relevance = judge_relevance(question, answer)
        judged_rows.append(
            {
                "faithful": judgment.faithful,
                "relevance": relevance,
                "is_unanswerable": is_unanswerable,
                "refused": looks_like_refusal(answer),
            }
        )

    retrieval_metrics = aggregate_retrieval_metrics(retrieval_rows)
    generation_metrics = aggregate_generation_metrics(judged_rows)
    latencies_sorted = sorted(latencies)
    p50 = statistics.median(latencies_sorted)
    p95 = latencies_sorted[min(int(len(latencies_sorted) * 0.95), len(latencies_sorted) - 1)]

    return {
        "config": config_name,
        "n_questions": len(eval_set),
        "retrieval": retrieval_metrics,
        "generation": generation_metrics,
        "latency_p50_s": round(p50, 3),
        "latency_p95_s": round(p95, 3),
        "avg_retry_count": round(sum(retry_counts) / len(retry_counts), 3) if retry_counts else 0,
        "cost_per_query_usd": 0.0,  # Groq free tier
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=list(ABLATION_CONFIGS.keys()))
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    eval_set = json.loads(EVAL_SET_PATH.read_text(encoding="utf-8"))

    configs = list(ABLATION_CONFIGS.keys()) if args.all else [args.config]
    if not configs or configs == [None]:
        parser.error("pass --config <name> or --all")

    app_config.EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}
    if RESULTS_PATH.exists():
        all_results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))

    for cfg in configs:
        print(f"Running config: {cfg} ...")
        result = run_config(cfg, eval_set)
        all_results[cfg] = result
        RESULTS_PATH.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
