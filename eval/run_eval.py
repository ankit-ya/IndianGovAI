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

from groq import RateLimitError
from langchain_core.messages import HumanMessage, SystemMessage

from src import config as app_config
from src.agent.graph import ask as agent_ask
from src.agent.nodes import GENERATE_SYSTEM_NO_TOOLS, format_context
from src.retrieval.pipeline import ABLATION_CONFIGS, retrieve as pipeline_retrieve
from eval.metrics import aggregate_generation_metrics, aggregate_retrieval_metrics, judge_answer, looks_like_refusal

EVAL_SET_PATH = app_config.EVAL_DIR / "eval_set.json"
RESULTS_PATH = app_config.EVAL_RESULTS_DIR / "ablation.json"


class DailyQuotaExhausted(Exception):
    pass


def _retry_on_rate_limit(fn, *args, max_attempts=3, **kwargs):
    """Groq's free tier has both per-minute and per-day token limits. Per-minute
    limits are worth a short sleep-and-retry; a per-day limit won't clear for a
    long time, so fail fast and let the caller stop the whole run cleanly instead
    of burning attempts against a wall."""
    for attempt in range(max_attempts):
        try:
            return fn(*args, **kwargs)
        except RateLimitError as e:
            msg = str(e)
            if "per day" in msg or "TPD" in msg or "RPD" in msg:
                raise DailyQuotaExhausted(msg) from e
            if attempt == max_attempts - 1:
                raise
            time.sleep(10 * (attempt + 1))


def _simple_generate(question: str, hits: list[dict]) -> str:
    """Non-agentic single-shot generation for the pure-retrieval ablation
    rows (no tools, no retry loop -- that's isolated to hybrid_rerank_agent)."""
    llm = app_config.get_chat_llm()
    context = format_context((h["chunk_id"], h["text"]) for h in hits)
    messages = [
        SystemMessage(content=GENERATE_SYSTEM_NO_TOOLS),
        HumanMessage(content=f"Question: {question}\n\nRetrieved context:\n{context or '(none)'}"),
    ]
    response = _retry_on_rate_limit(llm.invoke, messages)
    return response.content or ""


def run_config(config_name: str, eval_set: list[dict]) -> dict:
    retrieval_rows = []
    judged_rows = []
    latencies = []
    retry_counts = []
    errors = []

    for i, item in enumerate(eval_set, start=1):
        question = item["question"]
        is_unanswerable = item["type"] == "unanswerable"
        start = time.perf_counter()

        try:
            if config_name == "hybrid_rerank_agent":
                result = _retry_on_rate_limit(agent_ask, question)
                hits = [
                    {"chunk_id": d.metadata["chunk_id"], "doc_id": d.metadata["doc_id"], "text": d.page_content}
                    for d in result["documents"]
                ]
                answer = result.get("answer", "")
                retry_counts.append(result.get("retry_count", 0))
            else:
                hits = pipeline_retrieve(question, config_name, top_k=5)
                answer = _simple_generate(question, hits)
                retry_counts.append(0)
        except DailyQuotaExhausted:
            print(f"  [{i}/{len(eval_set)}] daily token quota exhausted, stopping this config early.")
            break
        except Exception as e:
            print(f"  [{i}/{len(eval_set)}] ERROR on question, skipping: {e}")
            errors.append({"question": question, "error": str(e)})
            continue

        elapsed = time.perf_counter() - start
        latencies.append(elapsed)

        retrieved_doc_ids = list(dict.fromkeys(h["doc_id"] for h in hits))
        retrieval_rows.append(
            {"retrieved_doc_ids": retrieved_doc_ids, "relevant_doc_ids": item["relevant_doc_ids"]}
        )

        context_text = format_context((h["chunk_id"], h.get("text", "")) for h in hits)
        try:
            judgment = _retry_on_rate_limit(judge_answer, question, context_text, answer)
        except DailyQuotaExhausted:
            print(f"  [{i}/{len(eval_set)}] daily token quota exhausted during judging, stopping this config early.")
            break
        judged_rows.append(
            {
                "faithful": judgment.faithful,
                "relevance": max(0.0, min(1.0, judgment.relevance_score)),
                "is_unanswerable": is_unanswerable,
                "refused": looks_like_refusal(answer),
            }
        )
        if i % 10 == 0:
            print(f"  [{i}/{len(eval_set)}] done")

    retrieval_metrics = aggregate_retrieval_metrics(retrieval_rows)
    generation_metrics = aggregate_generation_metrics(judged_rows)
    latencies_sorted = sorted(latencies)
    p50 = statistics.median(latencies_sorted) if latencies_sorted else None
    p95 = (
        latencies_sorted[min(int(len(latencies_sorted) * 0.95), len(latencies_sorted) - 1)]
        if latencies_sorted
        else None
    )

    return {
        "config": config_name,
        "n_questions": len(eval_set),
        "n_completed": len(judged_rows),
        "retrieval": retrieval_metrics,
        "generation": generation_metrics,
        "latency_p50_s": round(p50, 3) if p50 is not None else None,
        "latency_p95_s": round(p95, 3) if p95 is not None else None,
        "avg_retry_count": round(sum(retry_counts) / len(retry_counts), 3) if retry_counts else 0,
        "cost_per_query_usd": 0.0,  # Groq free tier
        "errors": errors,
    }


def stratified_sample(eval_set: list[dict], limit: int) -> list[dict]:
    """Proportionally sample across question types so a small --limit run still
    exercises single-hop, multi-hop, eligibility-reasoning, AND unanswerable
    questions -- a plain eval_set[:limit] would only grab single-hop questions
    (they're listed first) and completely miss refusal-behavior testing."""
    by_type: dict[str, list[dict]] = {}
    for item in eval_set:
        by_type.setdefault(item["type"], []).append(item)

    total = len(eval_set)
    sampled: list[dict] = []
    for qtype, items in by_type.items():
        share = max(1, round(limit * len(items) / total))
        sampled.extend(items[:share])
    return sampled[:limit] if len(sampled) > limit else sampled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=list(ABLATION_CONFIGS.keys()))
    parser.add_argument("--all", action="store_true")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Run on a small stratified subset instead of the full 50 questions (quick sanity check). "
        "Writes to ablation_smoketest.json instead of ablation.json.",
    )
    args = parser.parse_args()

    eval_set = json.loads(EVAL_SET_PATH.read_text(encoding="utf-8"))
    if args.limit:
        eval_set = stratified_sample(eval_set, args.limit)
        print(f"--limit {args.limit}: using {len(eval_set)} questions ({[e['type'] for e in eval_set]})")

    configs = list(ABLATION_CONFIGS.keys()) if args.all else [args.config]
    if not configs or configs == [None]:
        parser.error("pass --config <name> or --all")

    results_path = app_config.EVAL_RESULTS_DIR / ("ablation_smoketest.json" if args.limit else "ablation.json")
    app_config.EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}
    if results_path.exists():
        all_results = json.loads(results_path.read_text(encoding="utf-8"))

    for cfg in configs:
        print(f"Running config: {cfg} ...")
        result = run_config(cfg, eval_set)
        all_results[cfg] = result
        results_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
        if result["n_completed"] < result["n_questions"]:
            print(f"Config {cfg} stopped early ({result['n_completed']}/{result['n_questions']}) -- likely daily quota. Stopping run.")
            break


if __name__ == "__main__":
    main()
