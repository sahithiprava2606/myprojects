"""
Validation script for Phases 1-3 of the Transaction Anomaly Triage Agent.

Runs a battery of checks against fraud_data.db and the ChromaDB store,
prints PASS/FAIL with explanations for each, and gives an overall score.

Run AFTER you've completed generate_dataset.py, rules_engine.py, and vector_db.py:
    python validate.py
"""

import sqlite3
from pathlib import Path

DB_PATH = Path("fraud_data.db")
CHROMA_DIR = Path("chroma_db")

# Track results
results = []

def check(name, condition, detail="", warning=False):
    """Record a check result and print it."""
    if condition:
        status = "PASS"
        color = "\033[92m"  # green
    elif warning:
        status = "WARN"
        color = "\033[93m"  # yellow
    else:
        status = "FAIL"
        color = "\033[91m"  # red
    reset = "\033[0m"
    results.append((name, status))
    line = f"  {color}[{status}]{reset} {name}"
    if detail:
        line += f"\n         → {detail}"
    print(line)


def section(title):
    print(f"\n{'='*70}\n{title}\n{'='*70}")


# ===================================================================
# PHASE 1 — Data validation
# ===================================================================

def validate_phase_1():
    section("PHASE 1 — Dataset validation")

    if not DB_PATH.exists():
        check("Database file exists", False, f"{DB_PATH} not found. Run generate_dataset.py first.")
        return

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # --- Table presence and row counts ---
    print("\nTable row counts:")
    expected_ranges = {
        "customers":    (4_500, 5_500),
        "merchants":    (450, 550),
        "devices":      (5_500, 9_500),
        "transactions": (90_000, 110_000),
        "fraud_cases":  (50, 150),
    }
    for table, (lo, hi) in expected_ranges.items():
        try:
            count = cur.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            check(
                f"{table} row count in range [{lo:,}, {hi:,}]",
                lo <= count <= hi,
                f"actual: {count:,}",
            )
        except sqlite3.OperationalError as e:
            check(f"Table '{table}' exists", False, str(e))

    # --- Referential integrity ---
    print("\nReferential integrity:")
    orphan_txns_cust = cur.execute("""
        SELECT COUNT(*) FROM transactions t
        LEFT JOIN customers c ON t.customer_id = c.customer_id
        WHERE c.customer_id IS NULL
    """).fetchone()[0]
    check(
        "No transactions reference missing customers",
        orphan_txns_cust == 0,
        f"orphans found: {orphan_txns_cust}",
    )

    orphan_txns_merch = cur.execute("""
        SELECT COUNT(*) FROM transactions t
        LEFT JOIN merchants m ON t.merchant_id = m.merchant_id
        WHERE m.merchant_id IS NULL
    """).fetchone()[0]
    check(
        "No transactions reference missing merchants",
        orphan_txns_merch == 0,
        f"orphans found: {orphan_txns_merch}",
    )

    # --- Fraud distribution ---
    print("\nFraud distribution:")
    total = cur.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
    fraud = cur.execute("SELECT COUNT(*) FROM transactions WHERE is_fraud=1").fetchone()[0]
    fraud_pct = fraud / total * 100
    check(
        "Fraud rate between 1% and 6%",
        1.0 <= fraud_pct <= 6.0,
        f"actual: {fraud_pct:.2f}% ({fraud:,}/{total:,})",
    )

    # All five patterns present
    patterns = cur.execute(
        "SELECT fraud_pattern, COUNT(*) FROM transactions WHERE is_fraud=1 GROUP BY fraud_pattern"
    ).fetchall()
    pattern_dict = dict(patterns)
    expected_patterns = {"velocity", "geo_impossible", "amount_anomaly", "mcc_anomaly", "cnp_foreign"}
    missing = expected_patterns - set(pattern_dict.keys())
    check(
        "All 5 fraud patterns present",
        not missing,
        f"missing: {missing}" if missing else f"counts: {pattern_dict}",
    )

    # No single pattern dominates excessively (>80% would be suspicious)
    if pattern_dict:
        max_pattern_pct = max(pattern_dict.values()) / sum(pattern_dict.values()) * 100
        check(
            "No single pattern dominates (>80%)",
            max_pattern_pct < 80,
            f"largest pattern share: {max_pattern_pct:.1f}%",
            warning=True,
        )

    # --- Time coverage ---
    print("\nTemporal coverage:")
    min_ts, max_ts = cur.execute(
        "SELECT MIN(timestamp), MAX(timestamp) FROM transactions"
    ).fetchone()
    days_span = cur.execute(
        "SELECT CAST((JULIANDAY(MAX(timestamp)) - JULIANDAY(MIN(timestamp))) AS INTEGER) FROM transactions"
    ).fetchone()[0]
    check(
        "Transactions span 80-100 days",
        80 <= days_span <= 100,
        f"actual: {days_span} days ({min_ts} → {max_ts})",
    )

    # --- Customer-level sanity ---
    print("\nCustomer-level sanity:")
    avg_txns_per_cust = cur.execute(
        "SELECT AVG(c) FROM (SELECT COUNT(*) c FROM transactions GROUP BY customer_id)"
    ).fetchone()[0]
    check(
        "Average txns/customer between 10 and 40",
        10 <= avg_txns_per_cust <= 40,
        f"actual: {avg_txns_per_cust:.1f}",
    )

    conn.close()


# ===================================================================
# PHASE 2 — Rules engine validation
# ===================================================================

def validate_phase_2():
    section("PHASE 2 — Rules engine validation")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    try:
        alerts_count = cur.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
    except sqlite3.OperationalError:
        check("Alerts table exists", False, "Run rules_engine.py first.")
        conn.close()
        return

    # --- Alert volume ---
    print("\nAlert volume:")
    total_txns = cur.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
    alert_rate = alerts_count / total_txns * 100
    check(
        "Alert rate between 3% and 10%",
        3.0 <= alert_rate <= 10.0,
        f"actual: {alert_rate:.2f}% ({alerts_count:,}/{total_txns:,})",
    )

    # --- Precision and recall ---
    print("\nRules-engine precision & recall:")
    true_pos = cur.execute("SELECT COUNT(*) FROM alerts WHERE is_fraud=1").fetchone()[0]
    false_pos = alerts_count - true_pos
    total_fraud = cur.execute("SELECT COUNT(*) FROM transactions WHERE is_fraud=1").fetchone()[0]
    caught_fraud = cur.execute("""
        SELECT COUNT(DISTINCT t.transaction_id)
        FROM transactions t JOIN alerts a ON t.transaction_id = a.transaction_id
        WHERE t.is_fraud = 1
    """).fetchone()[0]

    precision = true_pos / alerts_count * 100 if alerts_count else 0
    recall = caught_fraud / total_fraud * 100 if total_fraud else 0

    check(
        "Precision between 30% and 70% (realistic noisy-rules range)",
        30 <= precision <= 70,
        f"actual: {precision:.2f}% (TP={true_pos:,}, FP={false_pos:,})",
    )
    check(
        "Recall above 90% (rules are wide net by design)",
        recall >= 90,
        f"actual: {recall:.2f}% (caught {caught_fraud:,}/{total_fraud:,} fraud txns)",
    )

    # --- Each rule fires at least sometimes ---
    print("\nPer-rule firing:")
    expected_rules = ["velocity", "geo_impossible", "amount_anomaly", "mcc_anomaly", "cnp_foreign"]
    for rule in expected_rules:
        count = cur.execute(
            "SELECT COUNT(*) FROM alerts WHERE triggered_rules LIKE ?",
            (f"%{rule}%",),
        ).fetchone()[0]
        check(
            f"Rule '{rule}' fires at least 50 times",
            count >= 50,
            f"actual: {count:,} alerts",
        )

    # --- Severity distribution ---
    print("\nSeverity distribution:")
    sev = dict(cur.execute("SELECT severity, COUNT(*) FROM alerts GROUP BY severity").fetchall())
    has_both = "high" in sev and "medium" in sev
    check(
        "Both 'high' and 'medium' severities present",
        has_both,
        f"counts: {sev}",
    )

    conn.close()


# ===================================================================
# PHASE 3 — Vector store validation
# ===================================================================

def validate_phase_3():
    section("PHASE 3 — Vector store validation")

    if not CHROMA_DIR.exists():
        check("ChromaDB directory exists", False, f"{CHROMA_DIR} not found. Run vector_db.py first.")
        return

    try:
        from vector_db import find_similar_cases, is_novel
    except ImportError:
        try:
            from vector_db import find_similar_cases, is_novel
        except ImportError:
            check("Can import vector store module", False,
                  "Could not import find_similar_cases. Check filename.")
            return

    # --- Store is populated ---
    import chromadb
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        collection = client.get_collection(name="fraud_cases")
        case_count = collection.count()
    except Exception as e:
        check("'fraud_cases' collection accessible", False, str(e))
        return

    print("\nVector store population:")
    check(
        "Vector store has 50-200 cases",
        50 <= case_count <= 200,
        f"actual: {case_count} cases",
    )

    # --- Retrieval quality: targeted queries should surface relevant patterns ---
    print("\nRetrieval quality — targeted queries:")
    targeted_queries = [
        # (description, expected_pattern_in_top3)
        ("Customer had 6 small online charges of $5-$20 each in 4 minutes from foreign IPs",
         "velocity"),
        ("Customer's typical spend is $400/month, sudden $3,500 electronics purchase",
         "amount_anomaly"),
        ("Card-not-present transaction from new foreign country, customer never went there",
         "cnp_foreign"),
        ("Customer who only buys groceries suddenly bought $1500 of cryptocurrency",
         "mcc_anomaly"),
        ("Genuine in-person purchase by customer in home country, followed by online purchase in foreign country within 30 minutes — physically impossible travel window between the two transactions",
         "geo_impossible"),
    ]

    correct = 0
    for desc, expected_pattern in targeted_queries:
        matches = find_similar_cases(desc, top_n=3)
        top_patterns = [m["pattern_tags"] for m in matches]
        match_found = expected_pattern in top_patterns
        top_sim = matches[0]["similarity"] if matches else 0
        check(
            f"Query about '{expected_pattern}' surfaces a matching case in top 3",
            match_found,
            f"top patterns: {top_patterns}, top similarity: {top_sim:.3f}",
        )
        if match_found:
            correct += 1

    accuracy = correct / len(targeted_queries) * 100
    check(
        "Overall retrieval accuracy ≥ 80% across targeted queries",
        accuracy >= 80,
        f"actual: {accuracy:.0f}% ({correct}/{len(targeted_queries)})",
    )

    # --- Novelty gate: should NOT flag relevant queries as novel ---
    print("\nNovelty gate — should NOT trigger on familiar patterns:")
    familiar_query = "Customer had multiple small online charges in 5 minutes from a foreign country"
    matches = find_similar_cases(familiar_query, top_n=3)
    novel = is_novel(matches)
    top_sim = max(m["similarity"] for m in matches) if matches else 0
    check(
        "Familiar pattern NOT flagged as novel",
        not novel,
        f"top similarity: {top_sim:.3f}, novel={novel}",
    )

    # --- Novelty gate: SHOULD flag truly novel scenarios ---
    print("\nNovelty gate — SHOULD trigger on unfamiliar patterns:")
    novel_query = "Customer received a phone call from someone claiming to be tech support, authorized wire transfer via voice using AI-generated voice deepfake of bank manager"
    matches = find_similar_cases(novel_query, top_n=3)
    novel = is_novel(matches)
    top_sim = max(m["similarity"] for m in matches) if matches else 0
    check(
        "Truly novel pattern flagged for human review",
        novel,
        f"top similarity: {top_sim:.3f}, novel={novel}",
        warning=True,  # Hard to guarantee, embeddings are fuzzy
    )

    # --- Similarity score distributions ---
    print("\nSimilarity score sanity:")
    # Use realistic alert-style descriptions (the kind the agent will actually generate),
    # not short toy phrases. Real RAG queries are 50-100 words with concrete details.
    sample_queries = [
        "Customer made 6 transactions of $5 to $25 each within 8 minutes at online retailers, all from IP addresses in a foreign country, card not present, no prior travel to that country",
        "Genuine in-person purchase in customer home country followed by online card-not-present transaction in a foreign country within 45 minutes, physically impossible travel window",
        "Charge of $4,200 at electronics retailer, more than 10x the customer's historical maximum transaction of approximately $350, customer's spending baseline is $400 per month",
        "First-ever transaction at a cryptocurrency exchange for a customer whose 18-month history shows only grocery, gas, and restaurant categories",
        "Online card-not-present transaction from a country the customer has never transacted with, customer home country is different, no record of international transactions in 24 months",
    ]
    all_top_sims = []
    for q in sample_queries:
        matches = find_similar_cases(q, top_n=1)
        if matches:
            all_top_sims.append(matches[0]["similarity"])
    avg_top_sim = sum(all_top_sims) / len(all_top_sims) if all_top_sims else 0
    check(
        "Average top similarity across queries between 0.45 and 0.85",
        0.45 <= avg_top_sim <= 0.85,
        f"actual: {avg_top_sim:.3f} across {len(all_top_sims)} queries",
    )


# ===================================================================
# Summary
# ===================================================================

def print_summary():
    section("VALIDATION SUMMARY")
    passed = sum(1 for _, s in results if s == "PASS")
    warned = sum(1 for _, s in results if s == "WARN")
    failed = sum(1 for _, s in results if s == "FAIL")
    total = len(results)

    print(f"\n  Total checks:  {total}")
    print(f"  \033[92mPASSED:\033[0m       {passed}")
    print(f"  \033[93mWARNINGS:\033[0m     {warned}")
    print(f"  \033[91mFAILED:\033[0m       {failed}")

    if failed > 0:
        print("\n  Failed checks:")
        for name, status in results:
            if status == "FAIL":
                print(f"    - {name}")

    print()
    if failed == 0:
        print("  All critical checks passed. Phases 1-3 are working as expected.")
        print("  Ready to proceed to Phase 4 (LangGraph agent).")
    elif failed <= 2:
        print("  A few minor issues. Review above and decide if they matter for your use case.")
    else:
        print("  Multiple failures — recommend addressing before moving to Phase 4.")


if __name__ == "__main__":
    print("Running validation for Phases 1-3...")
    validate_phase_1()
    validate_phase_2()
    validate_phase_3()
    print_summary()