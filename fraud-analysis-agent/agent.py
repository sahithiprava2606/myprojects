"""
Phase 4 — LangGraph agent for fraud alert triage.

For each alert, the agent runs through 6 nodes:
  1. fetch_context           — pull customer profile, history, AND nearby-window transactions
  2. compose_description     — build a natural-language summary with rule-specific evidence
  3. retrieve_similar_cases  — RAG query against ChromaDB
  4. novelty_gate            — if retrieval is weak, route directly to needs_review
  5. classify                — LLM call with structured output and anti-pattern guidance
  6. confidence_gate         — if LLM confidence is low, downgrade to needs_review

KEY CHANGE in this version (vs previous): velocity attacks were being misclassified as
false_positive because the LLM saw small amounts as "low risk" without realizing the
burst pattern was the fraud signature. We now fetch the burst window and surface its
statistics so the LLM has the actual evidence.
"""

import json
import os
import re
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Literal, Optional, TypedDict

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from openai import OpenAI
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END

from vector_db import find_similar_cases, is_novel, NOVELTY_THRESHOLD


# ---------- Configuration ----------
DB_PATH = Path("fraud_data.db")
MODEL = os.getenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-4")
CONFIDENCE_THRESHOLD = 0.70
CUSTOMER_HISTORY_DAYS = 90
BURST_WINDOW_MINUTES = 15   # how wide to fetch the nearby-transaction context

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)


# ---------- Structured output schema ----------
class Classification(BaseModel):
    classification: Literal["likely_fraud", "needs_review", "false_positive"] = Field(
        description="The triage decision for this alert"
    )
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    investigation_note: str


# ---------- Agent state ----------
class AgentState(TypedDict, total=False):
    alert_id: str
    transaction_id: str
    customer_id: str
    triggered_rules: str
    rule_details: str
    transaction: dict
    customer: dict
    history_summary: dict
    burst_window: List[dict]       # NEW — transactions in the surrounding time window
    description: str
    similar_cases: List[dict]
    max_similarity: float
    is_novel_pattern: bool
    classification: str
    confidence: float
    reasoning: str
    investigation_note: str
    route: str


# ---------- Helpers ----------
def safe_num(value, default=0.0) -> float:
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def safe_str(value, default="unknown") -> str:
    if value is None or (isinstance(value, str) and not value.strip()):
        return default
    return str(value)


def extract_json(text: str) -> str:
    if not text:
        return text
    text = text.strip()
    text = re.sub(r"^```(?:json|JSON)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else text


# ---------- Node 1: Fetch context (now also pulls burst window) ----------
def fetch_context(state: AgentState) -> AgentState:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # The flagged transaction
    txn = conn.execute("""
        SELECT t.*, m.merchant_name, m.mcc_category, m.country AS merchant_country, m.risk_tier
        FROM transactions t
        JOIN merchants m ON t.merchant_id = m.merchant_id
        WHERE t.transaction_id = ?
    """, (state["transaction_id"],)).fetchone()

    # Customer profile
    cust = conn.execute(
        "SELECT * FROM customers WHERE customer_id = ?", (state["customer_id"],)
    ).fetchone()

    txn_ts_str = txn["timestamp"]  # use SQLite's native string format directly

    # 90-day aggregate history (SQLite handles timestamp arithmetic in SQL to avoid
    # the Python isoformat-vs-SQLite-space-separator format mismatch)
    hist = conn.execute("""
        SELECT
            COUNT(*) AS txn_count,
            ROUND(AVG(t.amount), 2) AS avg_amount,
            ROUND(MAX(t.amount), 2) AS max_amount,
            COUNT(DISTINCT t.ip_country) AS distinct_ip_countries,
            GROUP_CONCAT(DISTINCT t.ip_country) AS ip_countries,
            COUNT(DISTINCT m.mcc_category) AS distinct_mccs,
            GROUP_CONCAT(DISTINCT m.mcc_category) AS mccs_used
        FROM transactions t
        JOIN merchants m ON t.merchant_id = m.merchant_id
        WHERE t.customer_id = ?
          AND t.timestamp >= datetime(?, ?)
          AND t.timestamp < ?
    """, (state["customer_id"], txn_ts_str, f'-{CUSTOMER_HISTORY_DAYS} days', txn_ts_str)).fetchone()

    # Burst window — same approach, SQLite handles the time arithmetic.
    # This was previously broken: Python's isoformat() uses 'T' separator but SQLite
    # stores timestamps with a space, so string comparison silently returned no rows.
    burst = conn.execute("""
        SELECT t.transaction_id, t.timestamp, t.amount, t.ip_country, t.card_present,
               m.merchant_name, m.mcc_category, m.country AS merchant_country
        FROM transactions t
        JOIN merchants m ON t.merchant_id = m.merchant_id
        WHERE t.customer_id = ?
          AND t.timestamp >= datetime(?, ?)
          AND t.timestamp <= datetime(?, ?)
        ORDER BY t.timestamp
    """, (state["customer_id"],
          txn_ts_str, f'-{BURST_WINDOW_MINUTES} minutes',
          txn_ts_str, f'+{BURST_WINDOW_MINUTES} minutes')).fetchall()

    conn.close()

    state["transaction"] = dict(txn)
    state["customer"] = dict(cust)
    state["history_summary"] = dict(hist) if hist else {}
    state["burst_window"] = [dict(b) for b in burst]
    return state


# ---------- Node 2: Compose description with rule-specific evidence ----------
def compose_description(state: AgentState) -> AgentState:
    txn = state["transaction"]
    cust = state["customer"]
    hist = state["history_summary"]
    rules = state["triggered_rules"]
    burst = state.get("burst_window", [])

    # Parse rule details
    try:
        rule_details_list = json.loads(state.get("rule_details", "[]"))
        rule_summary = "; ".join(r.get("detail", r.get("rule", "")) for r in rule_details_list)
    except (json.JSONDecodeError, TypeError):
        rule_summary = rules

    # Base alert description
    desc_parts = [
        f"FLAGGED TRANSACTION: ${safe_num(txn.get('amount')):.2f} at "
        f"{safe_str(txn.get('merchant_name'))} ({safe_str(txn.get('mcc_category'))}) "
        f"in {safe_str(txn.get('merchant_country'))}, "
        f"{'card-present' if txn.get('card_present') else 'card-not-present'}, "
        f"IP country {safe_str(txn.get('ip_country'))}, "
        f"timestamp {safe_str(txn.get('timestamp'))}.",

        f"CUSTOMER BASELINE: home country {safe_str(cust.get('home_country'))}, "
        f"avg monthly spend ${safe_num(cust.get('avg_monthly_spend')):.2f}, "
        f"historical max transaction ${safe_num(cust.get('max_historical_txn')):.2f}, "
        f"typical MCCs: {safe_str(cust.get('typical_mcc_list'), 'unknown')}.",

        f"90-DAY HISTORY: {int(safe_num(hist.get('txn_count')))} transactions, "
        f"avg ${safe_num(hist.get('avg_amount')):.2f}, "
        f"IP countries used: {safe_str(hist.get('ip_countries'), 'none')}, "
        f"MCCs used: {safe_str(hist.get('mccs_used'), 'none')}.",

        f"RULES TRIGGERED: {rules}. RULE DETAILS: {rule_summary}.",
    ]

    # CRITICAL: surface the burst window so velocity attacks are visible.
    # This is the new evidence the previous version was missing.
    if len(burst) >= 2:
        burst_amounts = [safe_num(b.get("amount")) for b in burst]
        burst_countries = sorted(set(safe_str(b.get("ip_country")) for b in burst))
        burst_merchants = sorted(set(safe_str(b.get("merchant_name")) for b in burst))
        first_ts = burst[0]["timestamp"]
        last_ts = burst[-1]["timestamp"]
        try:
            span_seconds = (datetime.fromisoformat(last_ts) - datetime.fromisoformat(first_ts)).total_seconds()
            span_str = f"{span_seconds/60:.1f} minutes" if span_seconds < 3600 else f"{span_seconds/3600:.1f} hours"
        except Exception:
            span_str = "unknown span"

        desc_parts.append(
            f"BURST-WINDOW EVIDENCE (±{BURST_WINDOW_MINUTES} min around alert): "
            f"{len(burst)} transactions in {span_str}, "
            f"amounts ${min(burst_amounts):.2f}-${max(burst_amounts):.2f} "
            f"(total ${sum(burst_amounts):.2f}), "
            f"across {len(burst_countries)} IP countries ({', '.join(burst_countries)}), "
            f"across {len(burst_merchants)} merchants. "
            f"NOTE: many small transactions in a short window from multiple foreign IPs is "
            f"the signature pattern of card-testing fraud, regardless of individual amounts."
        )
    else:
        desc_parts.append(
            f"BURST-WINDOW EVIDENCE (±{BURST_WINDOW_MINUTES} min around alert): "
            f"only this one transaction. No burst pattern present."
        )

    state["description"] = " ".join(desc_parts)
    return state


# ---------- Node 3: Retrieve similar cases ----------
def retrieve_similar_cases(state: AgentState) -> AgentState:
    matches = find_similar_cases(state["description"], top_n=3)
    state["similar_cases"] = matches
    state["max_similarity"] = max((m["similarity"] for m in matches), default=0.0)
    state["is_novel_pattern"] = is_novel(matches)
    return state


# ---------- Node 4: Novelty gate ----------
def novelty_gate(state: AgentState) -> AgentState:
    state["classification"] = "needs_review"
    state["confidence"] = 0.50
    state["reasoning"] = (
        f"Novel pattern detected — strongest retrieval similarity was "
        f"{state['max_similarity']:.3f}, below the novelty threshold of {NOVELTY_THRESHOLD}. "
        f"No strong historical precedent to ground classification."
    )
    state["investigation_note"] = (
        f"Alert flagged by rules: {state['triggered_rules']}. "
        f"No similar historical cases found in knowledge base (max similarity {state['max_similarity']:.2f}). "
        f"Recommending human review."
    )
    state["route"] = "novelty_gate"
    return state


# ---------- Node 5: Classify with rule-aware anti-patterns ----------
def classify(state: AgentState) -> AgentState:
    cases_text = "\n\n".join(
        f"Case {i+1} (similarity={c['similarity']:.2f}, outcome={c['outcome']}, "
        f"pattern={c['pattern_tags']}):\n{c['narrative']}"
        for i, c in enumerate(state["similar_cases"])
    )

    # Balanced prompt: explicit guidance for BOTH fraud and false_positive,
    # with anti-patterns scoped to require multiple co-occurring signals
    # (not just rule fired = fraud).
    system_prompt = (
        "You are a fraud-investigation agent triaging credit-card transaction alerts. "
        "Banks need real decisions. The rules engine that flagged this alert is INTENTIONALLY "
        "noisy — it catches 98% of fraud but ~50% of alerts are false alarms. Your job is to "
        "sort the real fraud from the legitimate customer behavior. BOTH classifications "
        "(likely_fraud AND false_positive) are equally valuable. Avoid defaulting either way.\n\n"

        "=== WHEN TO CLASSIFY AS likely_fraud ===\n"
        "Call likely_fraud when 2+ of these signals are TRUE TOGETHER:\n"
        "(a) BURST-WINDOW EVIDENCE shows 4+ transactions across multiple foreign IP countries "
        "or multiple merchants within ~15 minutes — the card-testing signature\n"
        "(b) The amount is significantly higher than the customer's 90-day max\n"
        "(c) The IP country is one the customer has NEVER transacted from before\n"
        "(d) The MCC is high-risk (crypto, gambling, money_transfer, jewelry) AND not in the "
        "customer's typical_mcc_list\n"
        "(e) Retrieved historical cases with outcome=confirmed_fraud have similarity ≥0.65 "
        "AND match the alert's specific pattern\n\n"

        "A SINGLE rule firing is NOT enough on its own — the rules are noisy by design. "
        "Look for multiple corroborating signals before classifying as fraud.\n\n"

        "=== CRITICAL: AMOUNT ANOMALY IS ITSELF A STRONG SIGNAL ===\n"
        "If a SINGLE transaction is more than 5x the customer's historical maximum, this is "
        "fraud-grade evidence by itself, regardless of whether a burst is present. Fraudsters "
        "using stolen cards for high-value goods (electronics, jewelry, gift cards) commit "
        "fraud in a single high-amount transaction — there is NO burst because they only "
        "need ONE successful charge. Examples that ARE likely_fraud:\n"
        "  - $10,000 charge when customer's max was $900 (11x) → likely_fraud\n"
        "  - $38,000 charge when customer's max was $3,000 (12x) → likely_fraud\n"
        "  - Even if the IP country is the customer's home country: if the amount is 5-10x "
        "the historical max with no plausible business explanation, it is FRAUD.\n"
        "Do NOT cite 'no burst pattern' as a reason to clear an amount-anomaly alert. "
        "A single huge transaction does not need a burst to be fraud.\n\n"

        "=== WHEN TO CLASSIFY AS false_positive ===\n"
        "Call false_positive when the evidence shows the alert is legitimate behavior. "
        "Specifically:\n"
        "(a) BURST-WINDOW EVIDENCE shows only 1 transaction or just 2-3 at the SAME merchant "
        "(split payment, not card testing)\n"
        "(b) The amount is within the customer's normal range "
        "(below or near their historical max)\n"
        "(c) The IP country, while foreign, IS in the customer's 90-day history (they've "
        "transacted from there before)\n"
        "(d) The MCC is in the customer's typical_mcc_list\n"
        "(e) Retrieved historical cases with outcome=false_positive have similarity ≥0.60 "
        "AND match this alert's pattern\n\n"

        "If the customer has used this country/merchant/MCC before, the rule firing is almost "
        "certainly noise. Don't be afraid to call false_positive — that is the system's purpose.\n\n"

        "=== WHEN TO USE needs_review ===\n"
        "Only when signals genuinely conflict (e.g., burst pattern present BUT all from same "
        "merchant, OR new country BUT amount is normal AND customer has international history). "
        "Do not use needs_review as a safe default.\n\n"

        "=== HOW TO REASON ===\n"
        "Walk through in order:\n"
        "1. Check BURST-WINDOW EVIDENCE — is it a card-testing burst (many merchants, "
        "many countries) or a split payment (same merchant)?\n"
        "2. Compare amount to customer's historical max\n"
        "3. Check if IP country appears in the customer's 90-day history\n"
        "4. Find the most similar retrieved case and use its outcome as your primary precedent\n"
        "5. Make the call. If the customer's history explains the rule firing, lean false_positive. "
        "If multiple signals point to fraud, lean likely_fraud.\n\n"

        "CONFIDENCE: 0.85+ for clear cases (3+ signals align), 0.70-0.84 for solid cases "
        "(2 signals align), below 0.70 → use needs_review.\n\n"

        "CRITICAL OUTPUT FORMAT: Your entire response must be a single valid JSON object. "
        "Do NOT wrap it in markdown code fences. Do NOT include any text before or after the JSON. "
        "Start your response with { and end with }."
    )

    user_prompt = (
        f"{state['description']}\n\n"
        f"TOP 3 SIMILAR HISTORICAL CASES:\n{cases_text}\n\n"
        f"Classify this alert. Work through the reasoning steps:\n"
        f"1. Burst-window evidence — card-testing burst, split payment, or single transaction?\n"
        f"2. Amount vs customer's historical max — anomalous or normal?\n"
        f"3. Is the IP country in the customer's 90-day history?\n"
        f"4. Most similar retrieved case — what was its outcome and how similar is it?\n"
        f"5. Make the call. If customer history explains the rule firing → false_positive. "
        f"If multiple fraud signals align → likely_fraud.\n\n"
        f"Respond with a single JSON object: "
        f"classification (\"likely_fraud\", \"needs_review\", or \"false_positive\"), "
        f"confidence (0-1), "
        f"reasoning (2-3 sentences citing the specific signals you checked above), "
        f"investigation_note (3 sentences for the analyst)."
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )

    raw = response.choices[0].message.content

    if not raw or not raw.strip():
        state["classification"] = "needs_review"
        state["confidence"] = 0.30
        state["reasoning"] = (
            f"LLM returned empty response (model={MODEL}). Provider may not fully support "
            f"response_format. Try a different model via OPENROUTER_MODEL env var."
        )
        state["investigation_note"] = "Agent received empty LLM response; recommending human review."
        state["route"] = "llm_empty"
        return state

    try:
        cleaned = extract_json(raw)
        parsed = Classification.model_validate_json(cleaned)
    except Exception as e:
        state["classification"] = "needs_review"
        state["confidence"] = 0.30
        state["reasoning"] = (
            f"LLM output parsing failed ({type(e).__name__}: {str(e)[:120]}). "
            f"Raw output (first 300 chars): {raw[:300]}"
        )
        state["investigation_note"] = "Agent encountered a parsing error; recommending human review."
        state["route"] = "llm_error"
        return state

    state["classification"] = parsed.classification
    state["confidence"] = parsed.confidence
    state["reasoning"] = parsed.reasoning
    state["investigation_note"] = parsed.investigation_note
    state["route"] = "llm"
    return state


# ---------- Node 6: Confidence gate ----------
def confidence_gate(state: AgentState) -> AgentState:
    if state["confidence"] < CONFIDENCE_THRESHOLD and state["classification"] != "needs_review":
        original = state["classification"]
        state["classification"] = "needs_review"
        state["reasoning"] += (
            f" [Downgraded from '{original}' — confidence {state['confidence']:.2f} "
            f"below threshold {CONFIDENCE_THRESHOLD}.]"
        )
        state["route"] = "confidence_gate"
    return state


def route_after_retrieval(state: AgentState) -> str:
    return "novelty_gate" if state["is_novel_pattern"] else "classify"


def build_agent():
    graph = StateGraph(AgentState)
    graph.add_node("fetch_context", fetch_context)
    graph.add_node("compose_description", compose_description)
    graph.add_node("retrieve_similar_cases", retrieve_similar_cases)
    graph.add_node("novelty_gate", novelty_gate)
    graph.add_node("classify", classify)
    graph.add_node("confidence_gate", confidence_gate)
    graph.set_entry_point("fetch_context")
    graph.add_edge("fetch_context", "compose_description")
    graph.add_edge("compose_description", "retrieve_similar_cases")
    graph.add_conditional_edges(
        "retrieve_similar_cases", route_after_retrieval,
        {"novelty_gate": "novelty_gate", "classify": "classify"},
    )
    graph.add_edge("novelty_gate", END)
    graph.add_edge("classify", "confidence_gate")
    graph.add_edge("confidence_gate", END)
    return graph.compile()


def triage_alert(alert: dict) -> dict:
    agent = build_agent()
    initial = {
        "alert_id": alert["alert_id"],
        "transaction_id": alert["transaction_id"],
        "customer_id": alert["customer_id"],
        "triggered_rules": alert["triggered_rules"],
        "rule_details": alert.get("rule_details", "[]"),
    }
    return agent.invoke(initial)


if __name__ == "__main__":
    if not os.getenv("OPENROUTER_API_KEY"):
        print("ERROR: Set OPENROUTER_API_KEY environment variable first.")
        exit(1)

    print(f"Using model: {MODEL}")
    # Smoke test on a velocity alert specifically — the previously-broken case
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    sample = conn.execute(
        "SELECT * FROM alerts WHERE is_fraud = 1 AND fraud_pattern = 'velocity' "
        "ORDER BY RANDOM() LIMIT 1"
    ).fetchone()
    if sample is None:
        sample = conn.execute("SELECT * FROM alerts ORDER BY RANDOM() LIMIT 1").fetchone()
    conn.close()

    print(f"\nAlert: {sample['alert_id']}")
    print(f"  Triggered rules: {sample['triggered_rules']}")
    print(f"  Ground truth (hidden from agent): is_fraud={sample['is_fraud']}, "
          f"pattern={sample['fraud_pattern']}")
    print("\nRunning agent...")

    result = triage_alert(dict(sample))

    print(f"\n--- Agent decision ---")
    print(f"Classification:    {result['classification']}")
    print(f"Confidence:        {result['confidence']:.2f}")
    print(f"Route taken:       {result['route']}")
    print(f"Max retrieval sim: {result['max_similarity']:.3f}")
    print(f"\nReasoning: {result['reasoning']}")
    print(f"\nInvestigation note: {result['investigation_note']}")