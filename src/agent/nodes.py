from __future__ import annotations

import re

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field

from src import config
from src.agent.state import GraphState, result_to_document
from src.agent.tools import TOOLS
from src.retrieval.pipeline import retrieve as pipeline_retrieve

RETRIEVAL_CONFIG = "hybrid_rerank_agent"
TOP_K = 5


def retrieve_node(state: GraphState) -> dict:
    query = state.get("rewritten_question") or state["question"]
    hits = pipeline_retrieve(query, RETRIEVAL_CONFIG, top_k=TOP_K)
    return {"documents": [result_to_document(h) for h in hits]}


class GradeResult(BaseModel):
    sufficient: bool = Field(description="True if the context can fully answer the question")
    reason: str = Field(description="One sentence on what's missing, if anything")


_GRADE_SYSTEM = (
    "You judge whether retrieved context is sufficient to answer a question about "
    "Indian government welfare schemes. Be strict: partial or tangential context is "
    "NOT sufficient. Respond with only compact JSON: "
    '{"sufficient": true|false, "reason": "..."}'
)


def grade_node(state: GraphState) -> dict:
    llm = config.get_chat_llm()
    context = "\n\n".join(f"[{d.metadata['chunk_id']}] {d.page_content}" for d in state["documents"])
    question = state.get("rewritten_question") or state["question"]
    prompt = f"Question: {question}\n\nRetrieved context:\n{context or '(no context retrieved)'}"

    # method="json_mode" (not the default forced-tool-call mode): the Groq-hosted
    # oss models here answer directly in JSON content rather than emitting a tool
    # call, which the default mode rejects with a 400 ("Tool choice is required,
    # but model did not call a tool") even though the JSON it produced was correct.
    structured = llm.with_structured_output(GradeResult, method="json_mode")
    try:
        result: GradeResult = structured.invoke(
            [SystemMessage(content=_GRADE_SYSTEM), HumanMessage(content=prompt)]
        )
        return {"grade_passed": result.sufficient, "grade_reason": result.reason}
    except Exception:
        # If structured parsing fails, fail closed (treat as insufficient) rather
        # than risk generating ungrounded from unjudged context.
        return {"grade_passed": False, "grade_reason": "grading call failed"}


_REWRITE_SYSTEM = (
    "You reformulate under-specified or poorly-matched questions about Indian "
    "government welfare schemes into a single, more retrievable search query. "
    "Expand acronyms, add scheme names if implied, and make implicit multi-hop "
    "questions explicit. Respond with ONLY the rewritten query, no explanation."
)


def rewrite_node(state: GraphState) -> dict:
    llm = config.get_chat_llm()
    original = state["question"]
    reason = state.get("grade_reason", "")
    prompt = f"Original question: {original}\nWhy retrieval failed: {reason}\n\nRewritten query:"
    response = llm.invoke([SystemMessage(content=_REWRITE_SYSTEM), HumanMessage(content=prompt)])
    rewritten = response.content.strip().strip('"')
    return {
        "rewritten_question": rewritten,
        "retry_count": state.get("retry_count", 0) + 1,
    }


_CITATION_RULE = (
    "Every factual claim must carry an inline citation using plain ASCII square "
    "brackets exactly like this: [chunk_id], matching a chunk_id from the context "
    "(not any other bracket style)."
)
_REFUSAL_RULE = (
    "If the context is insufficient to answer confidently, say so explicitly and "
    "do NOT guess or use outside knowledge."
)

# Used only where tools are actually bound (the agent's generate_node). A model
# told to "use tool X" will sometimes attempt a tool call even when no tools were
# bound to that particular request, which Groq's API then rejects outright -- see
# GENERATE_SYSTEM_NO_TOOLS below for the ablation-runner's tool-free variant.
GENERATE_SYSTEM = (
    "You are an assistant answering questions about Indian government welfare "
    "schemes (PMAY, Ayushman Bharat, PM-KISAN, and others) using ONLY the "
    f"retrieved context and tool outputs provided to you. Rules:\n1. {_CITATION_RULE}\n"
    f"2. {_REFUSAL_RULE}\n"
    "3. Use the eligibility_calculator, scheme_comparison, or document_lookup "
    "tools when the question needs rule application, cross-scheme comparison, "
    "or a specific section you don't already have.\n"
    "4. Keep answers concise and directly responsive."
)

# For the non-agent ablation configs (eval/run_eval.py's _simple_generate), which
# invoke the LLM directly with no tools bound -- omits the tool-use rule so the
# model doesn't attempt a tool call that isn't actually available on that request.
GENERATE_SYSTEM_NO_TOOLS = (
    "You are an assistant answering questions about Indian government welfare "
    "schemes (PMAY, Ayushman Bharat, PM-KISAN, and others) using ONLY the "
    f"retrieved context provided to you. Rules:\n1. {_CITATION_RULE}\n"
    f"2. {_REFUSAL_RULE}\n"
    "3. Keep answers concise and directly responsive."
)
_GENERATE_SYSTEM = GENERATE_SYSTEM  # backcompat alias within this module


# Accepts ASCII [chunk_id] as instructed, plus a couple of Unicode bracket
# styles (e.g. full-width 【chunk_id】) that Groq-hosted models sometimes use
# for citations regardless of instruction -- matching robustly here is cheaper
# than fighting model output formatting.
CITATION_RE = re.compile(r"[\[【(]([\w\-]+#c\d+)[\]】)]")
_CITATION_RE = CITATION_RE


def generate_node(state: GraphState) -> dict:
    llm = config.get_chat_llm().bind_tools(TOOLS)
    question = state.get("rewritten_question") or state["question"]
    context = "\n\n".join(f"[{d.metadata['chunk_id']}] {d.page_content}" for d in state["documents"])
    insufficient_note = ""
    if state.get("retry_count", 0) >= config.MAX_RETRIES and not state.get("grade_passed", True):
        insufficient_note = (
            "\n\nNOTE: Retrieval retried the maximum number of times and still could not "
            "find sufficient context. If you cannot answer confidently from what's below "
            "plus any tools, explicitly tell the user the information isn't available in "
            "the corpus rather than guessing."
        )

    messages = [
        SystemMessage(content=_GENERATE_SYSTEM),
        HumanMessage(
            content=f"Question: {question}\n\nRetrieved context:\n{context or '(none)'}{insufficient_note}"
        ),
    ]

    response = llm.invoke(messages)
    # Single round of tool execution if the model requested any tools.
    if getattr(response, "tool_calls", None):
        messages.append(response)
        tool_map = {t.name: t for t in TOOLS}
        for call in response.tool_calls:
            tool_fn = tool_map.get(call["name"])
            output = tool_fn.invoke(call["args"]) if tool_fn else f"Unknown tool: {call['name']}"
            messages.append(ToolMessage(content=str(output), tool_call_id=call["id"]))
        response = llm.invoke(messages)

    answer = response.content or ""
    citations = sorted(set(_CITATION_RE.findall(answer)))
    return {"answer": answer, "citations": citations}
