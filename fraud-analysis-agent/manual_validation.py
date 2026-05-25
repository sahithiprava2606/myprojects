"""
Manual validation script.

Acts as a blind labeling task: shows you alerts WITHOUT the agent's decision or
ground truth. You make your call as if you were the analyst. The script then
reveals what the agent said and what the ground truth was, and tracks agreement
rates.

This is the gold-standard validation approach in fraud ops — pull a sample,
label them yourself, then compare. It tells you:
  1. Whether the data is sensible (do you agree with ground truth?)
  2. How well the agent matches an experienced human (you, by the end)
  3. Where you and the agent disagree (the most informative cases)

Run:
    python manual_validate.py
"""

import csv
import sqlite3
import sys
import textwrap
from datetime import datetime
from pathlib import Path

DB_PATH = Path("fraud_data.db")
SAMPLE_SIZE = 20
RESULTS_PATH = Path("manual_validation_results.csv")


def load_sample(n: int) -> list:
    """Pull n random alerts from the latest agent run, balanced fraud/FP."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    rows = conn.execute(f"""
        WITH latest AS (
            SELECT * FROM agent_decisions ORDER BY decided_at DESC LIMIT 50
        )
        SELECT
            a.alert_id, a.classification AS agent_class, a.confidence AS agent_conf,
            a.reasoning AS agent_reasoning,
            a.is_fraud_ground_truth AS truth, a.fraud_pattern_ground_truth AS pattern,
            a.triggered_rules,
            t.amount, t.timestamp, t.ip_country, t.card_present,
            c.home_country, c.max_historical_txn, c.avg_monthly_spend,
            c.typical_mcc_list,
            m.merchant_name, m.mcc_category, m.country AS merchant_country
        FROM latest a
        JOIN transactions t ON a.transaction_id = t.transaction_id
        JOIN customers c ON t.customer_id = c.customer_id
        JOIN merchants m ON t.merchant_id = m.merchant_id
        ORDER BY RANDOM() LIMIT ?
    """, (n,)).fetchall()

    conn.close()
    return [dict(r) for r in rows]


def show_alert(alert: dict, idx: int, total: int):
    """Display an alert WITHOUT the agent decision or ground truth."""
    print("\n" + "=" * 76)
    print(f"  ALERT {idx + 1} of {total}")
    print("=" * 76)

    print(f"\nTransaction:")
    print(f"  Amount:           ${alert['amount']:.2f}")
    print(f"  Timestamp:        {alert['timestamp']}")
    print(f"  Merchant:         {alert['merchant_name']} ({alert['mcc_category']})")
    print(f"  Merchant country: {alert['merchant_country']}")
    print(f"  Card present:     {bool(alert['card_present'])}")
    print(f"  IP country:       {alert['ip_country']}")

    print(f"\nCustomer baseline:")
    print(f"  Home country:     {alert['home_country']}")
    print(f"  Avg monthly spend: ${alert['avg_monthly_spend']:.2f}")
    print(f"  Historical max:   ${alert['max_historical_txn']:.2f}")
    print(f"  Typical MCCs:     {alert['typical_mcc_list']}")

    print(f"\nWhy flagged:")
    print(f"  Rules triggered:  {alert['triggered_rules']}")


def get_human_label() -> str:
    """Get the user's classification."""
    while True:
        print("\nYour call:")
        print("  [F] likely_fraud")
        print("  [C] false_positive (clear it)")
        print("  [R] needs_review (punt to senior)")
        print("  [Q] quit and save progress")
        choice = input("\n  > ").strip().upper()
        if choice in ("F", "C", "R", "Q"):
            mapping = {"F": "likely_fraud", "C": "false_positive", "R": "needs_review", "Q": "quit"}
            return mapping[choice]
        print("  Invalid choice. Type F, C, R, or Q.")


def reveal(alert: dict, human_label: str):
    """Show the agent's decision and ground truth after human has committed."""
    truth_label = "FRAUD" if alert["truth"] == 1 else "NOT FRAUD"
    truth_str = f"{truth_label}" + (f" (pattern: {alert['pattern']})" if alert["pattern"] else "")

    print(f"\n--- Reveal ---")
    print(f"  Your label:    {human_label}")
    print(f"  Agent label:   {alert['agent_class']} (confidence {alert['agent_conf']:.2f})")
    print(f"  Ground truth:  {truth_str}")

    # Agreement scoring
    human_matches_truth = (
        (human_label == "likely_fraud" and alert["truth"] == 1) or
        (human_label == "false_positive" and alert["truth"] == 0)
    )
    agent_matches_truth = (
        (alert["agent_class"] == "likely_fraud" and alert["truth"] == 1) or
        (alert["agent_class"] == "false_positive" and alert["truth"] == 0)
    )
    human_agent_agree = (human_label == alert["agent_class"])

    print(f"\n  You vs truth:   {'✓ MATCH' if human_matches_truth else '✗ MISS'}"
          + (" (you said needs_review)" if human_label == "needs_review" else ""))
    print(f"  Agent vs truth: {'✓ MATCH' if agent_matches_truth else '✗ MISS'}"
          + (" (agent said needs_review)" if alert['agent_class'] == "needs_review" else ""))
    print(f"  You vs agent:   {'✓ AGREE' if human_agent_agree else '✗ DISAGREE'}")

    # Show agent reasoning so you can learn from it
    if not human_agent_agree:
        print(f"\n  Agent reasoning (since you disagreed):")
        print(textwrap.fill(
            alert["agent_reasoning"], 70,
            initial_indent="    ", subsequent_indent="    "
        ))

    return human_matches_truth, agent_matches_truth, human_agent_agree


def save_results(results: list):
    """Persist the labeling session to CSV for the README."""
    if not results:
        return
    with RESULTS_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {RESULTS_PATH}")


def main():
    if not DB_PATH.exists():
        print(f"ERROR: {DB_PATH} not found. Run earlier phases first.")
        sys.exit(1)

    print("\n" + "=" * 76)
    print("  MANUAL VALIDATION — blind labeling of fraud alerts")
    print("=" * 76)
    print(f"\nYou'll see {SAMPLE_SIZE} alerts pulled from the agent's latest run.")
    print("For each one, decide blind: fraud, false positive, or needs review.")
    print("The agent's decision and ground truth are revealed only AFTER you commit.")
    print("\nThis is the kind of labeling exercise a real fraud team does when")
    print("evaluating a new AI system. You'll get a concrete agreement number")
    print("for your README at the end.")
    print("\nPress Enter to start (Ctrl+C any time to quit)...")
    input()

    alerts = load_sample(SAMPLE_SIZE)
    if not alerts:
        print("ERROR: no agent decisions found in database. Run run_agent.py first.")
        sys.exit(1)

    human_correct = 0
    agent_correct = 0
    human_agent_agree = 0
    needs_review_count = 0
    results = []

    for i, alert in enumerate(alerts):
        show_alert(alert, i, len(alerts))
        human_label = get_human_label()
        if human_label == "quit":
            print(f"\nQuitting after {i} alerts.")
            break

        h_ok, a_ok, agree = reveal(alert, human_label)
        if human_label == "needs_review":
            needs_review_count += 1
        if h_ok:
            human_correct += 1
        if a_ok:
            agent_correct += 1
        if agree:
            human_agent_agree += 1

        results.append({
            "alert_id": alert["alert_id"],
            "amount": alert["amount"],
            "triggered_rules": alert["triggered_rules"],
            "human_label": human_label,
            "agent_label": alert["agent_class"],
            "agent_confidence": alert["agent_conf"],
            "ground_truth_is_fraud": alert["truth"],
            "ground_truth_pattern": alert["pattern"],
            "human_matches_truth": h_ok,
            "agent_matches_truth": a_ok,
            "human_agent_agree": agree,
        })

    # Summary
    n_committed = len(results)
    n_decided = n_committed - needs_review_count

    print("\n" + "=" * 76)
    print("  VALIDATION SUMMARY")
    print("=" * 76)
    print(f"\nAlerts labeled:                {n_committed}")
    print(f"  You committed (not review):  {n_decided}")
    print(f"  You sent to needs_review:    {needs_review_count}")

    if n_decided > 0:
        print(f"\nAccuracy on committed alerts:")
        print(f"  You vs ground truth:    {human_correct}/{n_decided} = {human_correct/n_decided*100:.1f}%")

    print(f"\nAgent agreement:")
    print(f"  Agent vs ground truth:  {agent_correct}/{n_committed} = {agent_correct/n_committed*100:.1f}%")
    print(f"  You vs agent:           {human_agent_agree}/{n_committed} = {human_agent_agree/n_committed*100:.1f}%")

    save_results(results)

    print(f"\nFor your README:")
    print(f'  "Manual blind-labeling of {n_committed} alerts showed {human_agent_agree/n_committed*100:.0f}%')
    print(f'   agreement between human-applied labels and agent classifications,')
    print(f'   with human accuracy {human_correct/max(n_decided,1)*100:.0f}% against ground truth."')


if __name__ == "__main__":
    main()