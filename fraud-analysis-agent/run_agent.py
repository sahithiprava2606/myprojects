"""
Phase 4 — Batch runner for the agent.

Pulls a sample of alerts from the database, runs each through the agent,
and writes results to both:
  - agent_decisions table in fraud_data.db (queryable for the dashboard)
  - agent_decisions.csv (easy to eyeball)

The sample is balanced: roughly half real fraud alerts and half false positives,
so you can see how the agent performs on both. This makes the metrics meaningful
on a small sample where pure random sampling would give all false positives
(since FP outnumber TP in the alerts table for some rule mixes).

Run:
    export OPENROUTER_API_KEY="sk-or-..."
    python run_agent.py

Override the sample size:
    python run_agent.py 100
"""

import csv
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

from agent import triage_alert, MODEL

DB_PATH = Path("fraud_data.db")
CSV_PATH = Path("agent_decisions.csv")
DEFAULT_SAMPLE_SIZE = 50


def fetch_balanced_sample(n: int) -> list:
    """
    Pull n alerts: half from real fraud, half from false positives.
    This ensures we exercise both code paths even on a small sample.
    """
    half = n // 2
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    fraud = conn.execute(
        "SELECT * FROM alerts WHERE is_fraud = 1 ORDER BY RANDOM() LIMIT ?",
        (half,),
    ).fetchall()
    non_fraud = conn.execute(
        "SELECT * FROM alerts WHERE is_fraud = 0 ORDER BY RANDOM() LIMIT ?",
        (n - half,),
    ).fetchall()
    conn.close()

    return [dict(a) for a in (list(fraud) + list(non_fraud))]


def setup_decisions_table():
    """Create the agent_decisions table if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS agent_decisions (
            alert_id TEXT PRIMARY KEY,
            transaction_id TEXT,
            customer_id TEXT,
            classification TEXT,
            confidence REAL,
            reasoning TEXT,
            investigation_note TEXT,
            route TEXT,
            max_similarity REAL,
            triggered_rules TEXT,
            is_fraud_ground_truth INTEGER,
            fraud_pattern_ground_truth TEXT,
            model TEXT,
            decided_at TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_decision(alert: dict, result: dict):
    """Persist one decision to both SQLite and the CSV (CSV appended row-by-row for crash safety)."""
    row = {
        "alert_id": alert["alert_id"],
        "transaction_id": alert["transaction_id"],
        "customer_id": alert["customer_id"],
        "classification": result["classification"],
        "confidence": round(result["confidence"], 4),
        "reasoning": result["reasoning"],
        "investigation_note": result["investigation_note"],
        "route": result["route"],
        "max_similarity": round(result["max_similarity"], 4),
        "triggered_rules": alert["triggered_rules"],
        "is_fraud_ground_truth": alert["is_fraud"],
        "fraud_pattern_ground_truth": alert.get("fraud_pattern") or "",
        "model": MODEL,
        "decided_at": datetime.utcnow().isoformat(timespec="seconds"),
    }

    # SQLite
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        f"INSERT OR REPLACE INTO agent_decisions ({','.join(row.keys())}) "
        f"VALUES ({','.join('?' for _ in row)})",
        tuple(row.values()),
    )
    conn.commit()
    conn.close()

    # CSV (append; write header if new file)
    write_header = not CSV_PATH.exists()
    with CSV_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def summarize(decisions: list, alerts: list):
    """Print a summary of how the agent performed vs ground truth."""
    print(f"\n{'='*70}\nAGENT RUN SUMMARY\n{'='*70}")
    print(f"Model: {MODEL}")
    print(f"Total alerts processed: {len(decisions)}")

    # Classification breakdown
    from collections import Counter
    class_counts = Counter(d["classification"] for d in decisions)
    print(f"\nClassifications:")
    for cls, ct in class_counts.most_common():
        print(f"  {cls:<20} {ct:>3}")

    # Route breakdown — how decisions were reached
    route_counts = Counter(d["route"] for d in decisions)
    print(f"\nDecision routes:")
    for r, ct in route_counts.most_common():
        print(f"  {r:<20} {ct:>3}")

    # The headline number — agreement with ground truth
    # We define: agent agrees if (likely_fraud + truth=fraud) OR (false_positive + truth=not fraud)
    # needs_review counts as "punted to human" — not right or wrong on its own
    agree = 0
    disagree = 0
    punted = 0
    for d, a in zip(decisions, alerts):
        truth_is_fraud = bool(a["is_fraud"])
        cls = d["classification"]
        if cls == "needs_review":
            punted += 1
        elif (cls == "likely_fraud" and truth_is_fraud) or (cls == "false_positive" and not truth_is_fraud):
            agree += 1
        else:
            disagree += 1

    decided = agree + disagree
    print(f"\nGround-truth comparison:")
    print(f"  Agent decided (not needs_review):  {decided}")
    print(f"    Agreed with ground truth:        {agree}")
    print(f"    Disagreed with ground truth:     {disagree}")
    if decided:
        print(f"    Accuracy on decided:             {agree/decided*100:.1f}%")
    print(f"  Punted to human (needs_review):    {punted}")

    # The interview-talking-points number: false-positive reduction
    fp_alerts = sum(1 for a in alerts if not a["is_fraud"])
    fp_correctly_cleared = sum(
        1 for d, a in zip(decisions, alerts)
        if not a["is_fraud"] and d["classification"] == "false_positive"
    )
    if fp_alerts:
        reduction = fp_correctly_cleared / fp_alerts * 100
        print(f"\nFalse-positive reduction:")
        print(f"  FP alerts in sample:        {fp_alerts}")
        print(f"  Correctly auto-cleared:     {fp_correctly_cleared}")
        print(f"  Workload reduction:         {reduction:.1f}%")

    # Dangerous-failure check: real fraud cleared as false_positive
    fraud_cleared = sum(
        1 for d, a in zip(decisions, alerts)
        if a["is_fraud"] and d["classification"] == "false_positive"
    )
    print(f"\nSAFETY CHECK — real fraud missed (classified as false_positive):")
    if fraud_cleared == 0:
        print(f"  None. Agent did not auto-clear any real fraud.")
    else:
        print(f"  {fraud_cleared} alerts. Inspect these in agent_decisions.csv:")
        for d, a in zip(decisions, alerts):
            if a["is_fraud"] and d["classification"] == "false_positive":
                print(f"    - {a['alert_id']} (pattern: {a.get('fraud_pattern')}, "
                      f"confidence: {d['confidence']:.2f})")


def main():
    import os
    if not os.getenv("OPENROUTER_API_KEY"):
        print("ERROR: Set OPENROUTER_API_KEY environment variable first.")
        print("  export OPENROUTER_API_KEY='sk-or-...'")
        sys.exit(1)

    sample_size = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SAMPLE_SIZE

    # Wipe previous CSV so we don't mix runs (SQLite uses INSERT OR REPLACE)
    if CSV_PATH.exists():
        CSV_PATH.unlink()
    setup_decisions_table()

    print(f"Fetching {sample_size} balanced alerts (half fraud, half FP)...")
    alerts = fetch_balanced_sample(sample_size)
    print(f"Running agent on {len(alerts)} alerts with model {MODEL}...")
    print(f"(Progress shown every 5 alerts. Errors will not stop the run.)\n")

    decisions = []
    start = time.time()
    for i, alert in enumerate(alerts, start=1):
        try:
            result = triage_alert(alert)
            decisions.append(result)
            save_decision(alert, result)
        except Exception as e:
            print(f"  [{i:>3}/{len(alerts)}] ERROR on {alert['alert_id']}: {type(e).__name__}: {e}")
            continue

        if i % 5 == 0:
            elapsed = time.time() - start
            rate = i / elapsed
            eta = (len(alerts) - i) / rate if rate > 0 else 0
            print(f"  [{i:>3}/{len(alerts)}] {alert['alert_id']} → "
                  f"{result['classification']} (conf {result['confidence']:.2f}, "
                  f"route {result['route']})  "
                  f"[{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining]")

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f}s ({elapsed/len(alerts):.1f}s/alert)")

    summarize(decisions, alerts[:len(decisions)])
    print(f"\nResults written to:")
    print(f"  - {DB_PATH} (table: agent_decisions)")
    print(f"  - {CSV_PATH}")


if __name__ == "__main__":
    main()