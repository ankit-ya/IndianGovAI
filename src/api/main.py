from __future__ import annotations

import json
import time
from pathlib import Path

from fastapi import FastAPI, Request
from pydantic import BaseModel

from src import config
from src.agent.graph import ask as agent_ask

app = FastAPI(title="Indian Gov Scheme RAG Assistant")

REQUEST_LOG = config.PROJECT_ROOT / "logs" / "requests.jsonl"


@app.middleware("http")
async def track_latency_and_cost(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    if request.url.path == "/query":
        REQUEST_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(REQUEST_LOG, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "ts": time.time(),
                        "path": request.url.path,
                        "latency_ms": round(elapsed_ms, 1),
                        "cost_usd": 0.0,  # Groq free tier
                    }
                )
                + "\n"
            )
    response.headers["X-Latency-Ms"] = str(round(elapsed_ms, 1))
    return response


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[str]
    retry_count: int
    latency_ms: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    start = time.perf_counter()
    result = agent_ask(req.question)
    latency_ms = (time.perf_counter() - start) * 1000
    return QueryResponse(
        answer=result.get("answer", ""),
        citations=result.get("citations", []),
        retry_count=result.get("retry_count", 0),
        latency_ms=round(latency_ms, 1),
    )
