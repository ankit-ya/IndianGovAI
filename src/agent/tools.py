"""Tool-calling functions bound to the generate node's LLM. Eligibility
thresholds in ELIGIBILITY_RULES are transcribed from the ingested guideline
PDFs (see the "source" field on each rule) rather than from model memory,
since scheme rules change and a stale hardcoded number is worse than
pointing back to the retrieved source text.
"""
from __future__ import annotations

from langchain_core.tools import tool

from src.ingestion.chunk import load_chunks
from src import config
from src.retrieval.pipeline import retrieve as pipeline_retrieve

# Filled in from the actual corpus after ingestion (see scripts/build_eligibility_rules.py
# note in README "Design decisions"). Each rule cites the chunk_id it was read from so an
# answer can point back to a verifiable source instead of asserting a bare number.
ELIGIBILITY_RULES: dict[str, dict] = {}


@tool
def eligibility_calculator(
    scheme: str,
    annual_income_inr: float | None = None,
    land_holding_acres: float | None = None,
    category: str | None = None,
) -> str:
    """Check basic eligibility for a named government scheme given household
    annual income in INR, land holding in acres, and social category
    (e.g. 'general', 'sc', 'st', 'obc', 'ews'). scheme should be one of the
    known scheme keys (pmay_urban, pmay_gramin, pm_kisan, ayushman_bharat).
    Returns a verdict plus the specific rule and source it was applied from.
    Falls back to returning the retrieved eligibility passages verbatim if no
    hardcoded rule is available for that scheme, so the caller can reason from
    source text directly instead of getting a wrong answer.
    """
    key = scheme.strip().lower().replace(" ", "_").replace("-", "_")
    rule = ELIGIBILITY_RULES.get(key)
    if not rule:
        hits = pipeline_retrieve(f"{scheme} eligibility criteria income land category", "hybrid_rerank_agent", top_k=3)
        if not hits:
            return f"No eligibility rule or source passages found for scheme '{scheme}'."
        passages = "\n---\n".join(f"[{h['chunk_id']}] {h['text'][:600]}" for h in hits)
        return (
            f"No hardcoded rule available for '{scheme}'. Retrieved eligibility passages "
            f"to reason from instead:\n{passages}"
        )

    checks = []
    verdict = "ELIGIBLE"
    if annual_income_inr is not None and "max_annual_income_inr" in rule:
        ok = annual_income_inr <= rule["max_annual_income_inr"]
        checks.append(f"income {annual_income_inr} <= {rule['max_annual_income_inr']}: {'OK' if ok else 'FAIL'}")
        verdict = verdict if ok else "NOT ELIGIBLE"
    if land_holding_acres is not None and "max_land_acres" in rule:
        ok = land_holding_acres <= rule["max_land_acres"]
        checks.append(f"land {land_holding_acres} <= {rule['max_land_acres']}: {'OK' if ok else 'FAIL'}")
        verdict = verdict if ok else "NOT ELIGIBLE"
    if category is not None and "eligible_categories" in rule:
        ok = category.strip().lower() in rule["eligible_categories"]
        checks.append(f"category '{category}' in {rule['eligible_categories']}: {'OK' if ok else 'FAIL'}")
        verdict = verdict if ok else "NOT ELIGIBLE"

    return (
        f"Scheme: {rule['scheme_name']}\nVerdict: {verdict}\n"
        f"Checks: {'; '.join(checks) if checks else 'no applicable criteria provided'}\n"
        f"Source: {rule['source_chunk_id']} ({rule['source_note']})"
    )


@tool
def scheme_comparison(schemes: list[str], attribute: str) -> str:
    """Retrieve raw source passages about a given attribute (e.g. 'income
    ceiling', 'subsidy amount', 'application process') for two or more named
    schemes side by side, to support comparing them. schemes should be scheme
    names as they appear in the corpus (e.g. 'PMAY-Urban', 'PM-KISAN')."""
    sections = []
    for scheme in schemes:
        hits = pipeline_retrieve(f"{scheme} {attribute}", "hybrid_rerank_agent", top_k=2)
        if hits:
            body = "\n".join(f"  [{h['chunk_id']}] {h['text'][:500]}" for h in hits)
        else:
            body = "  (no matching passages found)"
        sections.append(f"{scheme} — {attribute}:\n{body}")
    return "\n\n".join(sections)


@tool
def document_lookup(chunk_id: str) -> str:
    """Fetch the exact source text for a given chunk_id (format
    'doc_id#cN'), e.g. to verify a citation or pull full context around an
    already-retrieved answer."""
    doc_id = chunk_id.split("#")[0]
    for size in (512, 256):
        path = config.DATA_PROCESSED / f"chunks_{size}.json"
        if not path.exists():
            continue
        for c in load_chunks(path):
            if c.chunk_id == chunk_id:
                return f"[{c.chunk_id}] ({c.source_file}, p.{c.page_start}-{c.page_end})\n{c.text}"
    return f"chunk_id '{chunk_id}' not found (doc_id guess: {doc_id})."


TOOLS = [eligibility_calculator, scheme_comparison, document_lookup]
