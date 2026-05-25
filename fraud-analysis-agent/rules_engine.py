"""
Rules engine for the Transaction Anomaly Triage Agent.

Reads from fraud_data.db and produces an 'alerts' table.
Each rule is a function that takes (transaction, customer, customer_history)
and returns a dict describing the violation, or None.

Run AFTER generate_dataset.py:
    python rules_engine.py
"""

import json
import sqlite3
from datetime import timedelta
from pathlib import Path

import pandas as pd

DB_PATH = Path("fraud_data.db")

# Tunable thresholds — set conservatively to keep alert volume manageable
VELOCITY_WINDOW_MIN = 10
VELOCITY_MIN_COUNT = 4
GEO_IMPOSSIBLE_WINDOW_MIN = 60
AMOUNT_ANOMALY_MULTIPLIER = 5.0   # txn > 5x customer's historical max
HIGH_RISK_MCC = {"crypto", "gambling", "money_transfer", "jewelry"}


# ---------- Individual rules ----------

def rule_velocity(txn, customer, history):
    """N+ transactions within a short window — card-testing signature."""
    window_start = txn["timestamp"] - timedelta(minutes=VELOCITY_WINDOW_MIN)
    recent = history[
        (history["timestamp"] >= window_start) &
        (history["timestamp"] <= txn["timestamp"])
    ]
    if len(recent) >= VELOCITY_MIN_COUNT:
        return {
            "rule": "velocity",
            "severity": "high",
            "detail": f"{len(recent)} transactions in last {VELOCITY_WINDOW_MIN} minutes",
        }
    return None


def rule_geo_impossible(txn, customer, history):
    """A prior transaction in a different country within an impossible travel window."""
    window_start = txn["timestamp"] - timedelta(minutes=GEO_IMPOSSIBLE_WINDOW_MIN)
    recent = history[
        (history["timestamp"] >= window_start) &
        (history["timestamp"] < txn["timestamp"])
    ]
    if recent.empty:
        return None

    prior_countries = set(recent["ip_country"].dropna().unique())
    prior_countries.discard(txn["ip_country"])
    if prior_countries:
        return {
            "rule": "geo_impossible",
            "severity": "high",
            "detail": f"prior txn in {sorted(prior_countries)} within {GEO_IMPOSSIBLE_WINDOW_MIN} min",
        }
    return None


def rule_amount_anomaly(txn, customer, history):
    """Transaction far exceeds the customer's historical max."""
    threshold = customer["max_historical_txn"] * AMOUNT_ANOMALY_MULTIPLIER
    if txn["amount"] > threshold:
        return {
            "rule": "amount_anomaly",
            "severity": "high",
            "detail": f"${txn['amount']:.2f} is {txn['amount']/customer['max_historical_txn']:.1f}x historical max (${customer['max_historical_txn']:.2f})",
        }
    return None


def rule_mcc_anomaly(txn, customer, history):
    """High-risk merchant category the customer has never used."""
    if txn["mcc_category"] not in HIGH_RISK_MCC:
        return None
    typical = set(customer["typical_mcc_list"].split(","))
    if txn["mcc_category"] not in typical:
        # Also check they've never had any transaction in this MCC
        prior_mccs = set(history["mcc_category"].unique()) if not history.empty else set()
        if txn["mcc_category"] not in prior_mccs:
            return {
                "rule": "mcc_anomaly",
                "severity": "medium",
                "detail": f"first-ever transaction in high-risk category '{txn['mcc_category']}'",
            }
    return None


def rule_cnp_foreign(txn, customer, history):
    """Card-not-present in a country the customer has no transaction history with."""
    if txn["card_present"]:
        return None
    if txn["ip_country"] == customer["home_country"]:
        return None
    # Has the customer ever transacted from this country before?
    prior_ip_countries = set(history["ip_country"].dropna().unique()) if not history.empty else set()
    if txn["ip_country"] not in prior_ip_countries:
        return {
            "rule": "cnp_foreign",
            "severity": "medium",
            "detail": f"CNP transaction from new country '{txn['ip_country']}' (home: {customer['home_country']})",
        }
    return None


RULES = [rule_velocity, rule_geo_impossible, rule_amount_anomaly, rule_mcc_anomaly, rule_cnp_foreign]


# ---------- Engine ----------

def run_engine():
    print(f"Loading data from {DB_PATH}...")
    conn = sqlite3.connect(DB_PATH)

    # Pull everything once into memory — 100k rows is small enough
    customers = pd.read_sql("SELECT * FROM customers", conn).set_index("customer_id")
    merchants = pd.read_sql("SELECT * FROM merchants", conn).set_index("merchant_id")
    txns = pd.read_sql("SELECT * FROM transactions", conn, parse_dates=["timestamp"])

    # Enrich transactions with merchant info (so rules can see mcc_category)
    txns = txns.merge(
        merchants[["mcc_category", "country", "risk_tier"]].rename(columns={"country": "merchant_country"}),
        left_on="merchant_id", right_index=True, how="left"
    )
    txns = txns.sort_values("timestamp").reset_index(drop=True)

    # Pre-group transactions by customer for fast history lookups
    print(f"Indexing {len(txns):,} transactions by customer...")
    txns_by_customer = {cid: g for cid, g in txns.groupby("customer_id")}

    print(f"Scanning with {len(RULES)} rules...")
    alerts = []
    for i, txn in enumerate(txns.itertuples(index=False), start=1):
        txn_dict = txn._asdict()
        try:
            customer = customers.loc[txn_dict["customer_id"]]
        except KeyError:
            continue

        history = txns_by_customer.get(txn_dict["customer_id"], pd.DataFrame())
        # History = transactions strictly BEFORE this one (so the current txn isn't its own context)
        history = history[history["timestamp"] < txn_dict["timestamp"]]

        triggered = []
        for rule_fn in RULES:
            result = rule_fn(txn_dict, customer, history)
            if result:
                triggered.append(result)

        if triggered:
            severity = "high" if any(r["severity"] == "high" for r in triggered) else "medium"
            alerts.append({
                "alert_id": f"ALERT_{i:08d}",
                "transaction_id": txn_dict["transaction_id"],
                "customer_id": txn_dict["customer_id"],
                "timestamp": txn_dict["timestamp"],
                "amount": txn_dict["amount"],
                "severity": severity,
                "triggered_rules": ",".join(r["rule"] for r in triggered),
                "rule_details": json.dumps(triggered),
                "is_fraud": txn_dict["is_fraud"],
                "fraud_pattern": txn_dict["fraud_pattern"],
                "status": "pending",  # for human review later
            })

        if i % 20000 == 0:
            print(f"  scanned {i:,}/{len(txns):,} — {len(alerts):,} alerts so far")

    alerts_df = pd.DataFrame(alerts)

    # Persist
    print(f"\nWriting {len(alerts_df):,} alerts to database...")
    conn.execute("DROP TABLE IF EXISTS alerts")
    alerts_df.to_sql("alerts", conn, index=False)
    conn.execute("CREATE INDEX idx_alerts_status ON alerts(status)")
    conn.execute("CREATE INDEX idx_alerts_severity ON alerts(severity)")
    conn.commit()

    # ---- Summary: the headline metrics ----
    print("\n=== Rules engine summary ===")
    total_txns = len(txns)
    total_alerts = len(alerts_df)
    print(f"Total transactions:   {total_txns:>8,}")
    print(f"Total alerts:         {total_alerts:>8,}  ({total_alerts/total_txns*100:.2f}% of txns)")

    true_pos = alerts_df["is_fraud"].sum()
    false_pos = total_alerts - true_pos
    print(f"  true positives:     {true_pos:>8,}  (actually fraud)")
    print(f"  false positives:    {false_pos:>8,}  (look suspicious, aren't)")
    print(f"  precision:          {true_pos/total_alerts*100:>7.2f}%")

    total_fraud = txns["is_fraud"].sum()
    caught = alerts_df["is_fraud"].sum()
    print(f"  fraud recall:       {caught/total_fraud*100:>7.2f}%  ({caught:,}/{total_fraud:,} fraud txns caught)")

    print("\nAlerts by triggered rule:")
    rule_counts = {}
    for rules_str in alerts_df["triggered_rules"]:
        for r in rules_str.split(","):
            rule_counts[r] = rule_counts.get(r, 0) + 1
    for r, c in sorted(rule_counts.items(), key=lambda x: -x[1]):
        print(f"  {r:<18} {c:>6,}")

    print("\nAlerts by severity:")
    print(alerts_df["severity"].value_counts().to_string())

    conn.close()
    print(f"\nDone. The 'alerts' table is the input queue for the agent.")


if __name__ == "__main__":
    run_engine()