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

ACRE_TO_HECTARE = 0.404686

# Transcribed directly from the ingested PDFs (see source_doc_id/source_page on each
# rule -- run document_lookup or open the PDF at that page to verify). Ayushman
# Bharat/PM-JAY is deliberately absent: its eligibility runs off SECC 2011
# deprivation/occupational criteria, not a numeric income or land threshold, so a
# hardcoded rule for it would misrepresent how the scheme actually works -- it falls
# through to the retrieval fallback below instead.
ELIGIBILITY_RULES: dict[str, dict] = {
    "pm_kisan": {
        "scheme_name": "PM-KISAN",
        "max_land_hectares": 2.0,
        "benefit": "Rs. 6,000/year per family, in 3 installments of Rs. 2,000 (land records cutoff 01.02.2019)",
        "caveats": (
            "Land-holding is the sole numeric criterion; families are also excluded if any "
            "member is a current/former constitutional post holder, current/former MP/MLA, "
            "serving/retired government or PSU employee (excluding Group D/MTS), a pensioner "
            "drawing >= Rs.10,000/month, an income-tax payer in the last assessment year, or a "
            "registered professional (doctor/engineer/lawyer/CA/architect) in practice."
        ),
        "source_note": "PM-KISAN FAQ, Q3/Q6/Q11/Q12",
        "source_doc_id": "pmkisan_faq",
        "source_page": 1,
    },
    "pmay_urban_ews": {
        "scheme_name": "PMAY-Urban CLSS (EWS)",
        "max_annual_income_inr": 300000,
        "benefit": "6.5% interest subsidy on home loans up to Rs. 6 lakh; carpet area up to 30 sq.m.",
        "caveats": "Beneficiary family must not already own a pucca house anywhere in India.",
        "source_note": "CLSS for EWS/LIG Operational Guidelines, Definitions section",
        "source_doc_id": "pmay_urban_clss_ews_lig_guidelines",
        "source_page": 6,
    },
    "pmay_urban_lig": {
        "scheme_name": "PMAY-Urban CLSS (LIG)",
        "min_annual_income_inr": 300001,
        "max_annual_income_inr": 600000,
        "benefit": "6.5% interest subsidy on home loans up to Rs. 6 lakh; carpet area up to 60 sq.m.",
        "caveats": "Beneficiary family must not already own a pucca house anywhere in India.",
        "source_note": "CLSS for EWS/LIG Operational Guidelines, Definitions section",
        "source_doc_id": "pmay_urban_clss_ews_lig_guidelines",
        "source_page": 6,
    },
    "west_bengal_krishak_bandhu": {
        "scheme_name": "West Bengal Krishak Bandhu",
        "benefit": (
            "Rs. 10,000/year (2 installments) for farmers with >=1 acre cultivable land; "
            "pro-rata down to a minimum of Rs. 4,000/year for less than 1 acre"
        ),
        "source_note": "Krishak Bandhu FAQ, Q4",
        "source_doc_id": "west_bengal_krishak_bandhu_faq",
        "source_page": 5,
    },
}


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

    if annual_income_inr is not None:
        if "max_annual_income_inr" in rule:
            ok = annual_income_inr <= rule["max_annual_income_inr"]
            checks.append(f"income {annual_income_inr} <= {rule['max_annual_income_inr']}: {'OK' if ok else 'FAIL'}")
            verdict = verdict if ok else "NOT ELIGIBLE"
        if "min_annual_income_inr" in rule:
            ok = annual_income_inr >= rule["min_annual_income_inr"]
            checks.append(f"income {annual_income_inr} >= {rule['min_annual_income_inr']}: {'OK' if ok else 'FAIL'}")
            verdict = verdict if ok else "NOT ELIGIBLE"
    if land_holding_acres is not None and "max_land_hectares" in rule:
        land_hectares = land_holding_acres * ACRE_TO_HECTARE
        ok = land_hectares <= rule["max_land_hectares"]
        checks.append(
            f"land {land_holding_acres} acres ({land_hectares:.2f} ha) <= "
            f"{rule['max_land_hectares']} ha: {'OK' if ok else 'FAIL'}"
        )
        verdict = verdict if ok else "NOT ELIGIBLE"
    if category is not None and "eligible_categories" in rule:
        ok = category.strip().lower() in rule["eligible_categories"]
        checks.append(f"category '{category}' in {rule['eligible_categories']}: {'OK' if ok else 'FAIL'}")
        verdict = verdict if ok else "NOT ELIGIBLE"

    caveats = f"\nCaveats: {rule['caveats']}" if "caveats" in rule else ""
    return (
        f"Scheme: {rule['scheme_name']}\nVerdict: {verdict}\n"
        f"Benefit if eligible: {rule['benefit']}\n"
        f"Checks: {'; '.join(checks) if checks else 'no applicable criteria provided'}"
        f"{caveats}\n"
        f"Source: {rule['source_doc_id']} p.{rule['source_page']} ({rule['source_note']})"
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
