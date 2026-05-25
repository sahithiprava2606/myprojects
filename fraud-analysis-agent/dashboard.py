"""
Phase 5 — Streamlit dashboard for the Transaction Anomaly Triage Agent.

Four pages:
  1. Alert Queue        — sortable list of all agent-triaged alerts
  2. Alert Detail       — drill into one alert, see reasoning, confirm/override
  3. Analytics          — KPIs, charts, confusion matrix
  4. Pattern Explorer   — query the RAG vector store directly

The "Confirm Fraud" button in Alert Detail closes the human-in-the-loop:
  - Records the analyst's decision
  - Auto-adds the confirmed fraud as a narrative to the vector store
  - Next similar alert benefits from this knowledge

Run:
    pip install streamlit plotly
    streamlit run dashboard.py
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from vector_db import find_similar_cases, add_case_from_alert


DB_PATH = Path("fraud_data.db")

# ==================== Page config ====================
st.set_page_config(
    page_title="Fraud Triage Agent",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==================== Database setup ====================

def setup_analyst_tracking():
    """Add columns and table needed for human-in-the-loop tracking (idempotent)."""
    conn = sqlite3.connect(DB_PATH)

    # Add analyst_decision column to agent_decisions if not present
    cols = [r[1] for r in conn.execute("PRAGMA table_info(agent_decisions)").fetchall()]
    if "analyst_decision" not in cols:
        conn.execute("ALTER TABLE agent_decisions ADD COLUMN analyst_decision TEXT")
        conn.execute("ALTER TABLE agent_decisions ADD COLUMN analyst_decided_at TEXT")
        conn.execute("ALTER TABLE agent_decisions ADD COLUMN analyst_note TEXT")

    conn.commit()
    conn.close()


setup_analyst_tracking()


# ==================== Data loading (cached) ====================

@st.cache_data(ttl=10)  # refresh every 10s so analyst decisions show up quickly
def load_decisions() -> pd.DataFrame:
    """Load all agent decisions with joined transaction and customer data."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("""
        SELECT
            a.alert_id, a.transaction_id, a.customer_id,
            a.classification, a.confidence, a.reasoning, a.investigation_note,
            a.route, a.max_similarity, a.triggered_rules,
            a.is_fraud_ground_truth, a.fraud_pattern_ground_truth,
            a.model, a.decided_at,
            a.analyst_decision, a.analyst_decided_at, a.analyst_note,
            t.amount, t.timestamp AS txn_timestamp, t.ip_country, t.card_present,
            c.home_country, c.max_historical_txn, c.avg_monthly_spend, c.typical_mcc_list,
            m.merchant_name, m.mcc_category, m.country AS merchant_country, m.risk_tier
        FROM agent_decisions a
        JOIN transactions t ON a.transaction_id = t.transaction_id
        JOIN customers c ON a.customer_id = c.customer_id
        JOIN merchants m ON t.merchant_id = m.merchant_id
        ORDER BY a.decided_at DESC
    """, conn)
    conn.close()
    return df


def record_analyst_decision(alert_id: str, decision: str, note: str = ""):
    """Persist an analyst's decision back to the database."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "UPDATE agent_decisions SET analyst_decision = ?, analyst_decided_at = ?, analyst_note = ? "
        "WHERE alert_id = ?",
        (decision, datetime.utcnow().isoformat(timespec="seconds"), note, alert_id),
    )
    conn.commit()
    conn.close()
    st.cache_data.clear()


# ==================== Sidebar nav ====================

st.sidebar.title("🛡️ Fraud Triage")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Page",
    ["📋 Alert Queue", "🔍 Alert Detail", "📊 Analytics", "🔎 Pattern Explorer"],
    label_visibility="collapsed",
)

# Quick stats in sidebar
df_all = load_decisions()
st.sidebar.markdown("---")
st.sidebar.markdown("**At a glance**")
st.sidebar.metric("Total alerts triaged", len(df_all))
st.sidebar.metric("Pending analyst review", int(df_all["analyst_decision"].isna().sum()))
fraud_count = int((df_all["classification"] == "likely_fraud").sum())
st.sidebar.metric("Flagged as likely fraud", fraud_count)


# ==================== Helpers ====================

def class_badge(classification: str) -> str:
    """Return an HTML-styled badge for a classification."""
    colors = {
        "likely_fraud":   ("#ffe2e2", "#c92a2a"),
        "needs_review":   ("#fff4d6", "#a17008"),
        "false_positive": ("#d3f9d8", "#2b8a3e"),
    }
    bg, fg = colors.get(classification, ("#e9ecef", "#495057"))
    return (
        f'<span style="background:{bg};color:{fg};padding:3px 10px;'
        f'border-radius:12px;font-size:13px;font-weight:600">{classification}</span>'
    )


# =====================================================================
# PAGE 1 — Alert Queue
# =====================================================================

if page == "📋 Alert Queue":
    st.title("Alert Queue")
    st.caption("All alerts the agent has triaged. Click an alert ID below to drill in.")

    col1, col2, col3 = st.columns(3)
    with col1:
        class_filter = st.multiselect(
            "Classification",
            options=df_all["classification"].dropna().unique().tolist(),
            default=df_all["classification"].dropna().unique().tolist(),
        )
    with col2:
        review_filter = st.selectbox(
            "Analyst review status",
            options=["All", "Pending review", "Reviewed"],
        )
    with col3:
        min_conf = st.slider("Min confidence", 0.0, 1.0, 0.0, step=0.05)

    df = df_all[df_all["classification"].isin(class_filter)].copy()
    if review_filter == "Pending review":
        df = df[df["analyst_decision"].isna()]
    elif review_filter == "Reviewed":
        df = df[df["analyst_decision"].notna()]
    df = df[df["confidence"] >= min_conf]

    st.markdown(f"**{len(df)} alerts** match your filters.")

    # Compact queue view — most important columns only
    display = df[[
        "alert_id", "classification", "confidence", "amount", "merchant_name",
        "ip_country", "home_country", "triggered_rules", "analyst_decision",
    ]].rename(columns={
        "alert_id": "Alert",
        "classification": "Agent call",
        "confidence": "Conf",
        "amount": "Amount",
        "merchant_name": "Merchant",
        "ip_country": "IP",
        "home_country": "Home",
        "triggered_rules": "Rules",
        "analyst_decision": "Analyst",
    })

    # Color-code rows by classification
    def color_row(row):
        c = row["Agent call"]
        if c == "likely_fraud":
            return ["background-color: #fff5f5"] * len(row)
        elif c == "false_positive":
            return ["background-color: #f3f9f4"] * len(row)
        elif c == "needs_review":
            return ["background-color: #fffbf0"] * len(row)
        return [""] * len(row)

    styled = display.style.apply(color_row, axis=1).format({"Conf": "{:.2f}", "Amount": "${:.2f}"})
    st.dataframe(styled, use_container_width=True, height=600)

    st.markdown("---")
    st.info("To drill into an alert, copy its ID and paste it into the **Alert Detail** page.")


# =====================================================================
# PAGE 2 — Alert Detail (human-in-the-loop)
# =====================================================================

elif page == "🔍 Alert Detail":
    st.title("Alert Detail")
    st.caption("Drill into a single alert. Confirm or override the agent's classification.")

    # Alert picker — sort by pending-review first, then by confidence asc (most uncertain first)
    df_pending = df_all[df_all["analyst_decision"].isna()].sort_values("confidence")
    df_done = df_all[df_all["analyst_decision"].notna()]
    options = (
        [f"⏳ {r.alert_id} — {r.classification} (conf {r.confidence:.2f}) — ${r.amount:.0f}"
         for _, r in df_pending.iterrows()]
        +
        [f"✓ {r.alert_id} — {r.classification} → analyst: {r.analyst_decision}"
         for _, r in df_done.iterrows()]
    )
    if not options:
        st.warning("No alerts in the database. Run `python run_agent.py` first.")
        st.stop()

    choice = st.selectbox("Select an alert", options)
    alert_id = choice.split(" ")[1]
    row = df_all[df_all["alert_id"] == alert_id].iloc[0]

    # --- Header card ---
    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        st.markdown(f"### {row['alert_id']}")
        st.markdown(f"Agent classification: {class_badge(row['classification'])}", unsafe_allow_html=True)
    with c2:
        st.metric("Agent confidence", f"{row['confidence']:.2f}")
    with c3:
        st.metric("Retrieval similarity", f"{row['max_similarity']:.2f}")

    st.markdown("---")

    # --- Transaction + customer context ---
    col_txn, col_cust = st.columns(2)
    with col_txn:
        st.subheader("Transaction")
        st.markdown(f"**Amount**: ${row['amount']:.2f}")
        st.markdown(f"**Timestamp**: {row['txn_timestamp']}")
        st.markdown(f"**Merchant**: {row['merchant_name']} *({row['mcc_category']})*")
        st.markdown(f"**Merchant country**: {row['merchant_country']}  &nbsp;**Risk tier**: {row['risk_tier']}")
        st.markdown(f"**IP country**: {row['ip_country']}  &nbsp;**Card present**: {bool(row['card_present'])}")
    with col_cust:
        st.subheader("Customer baseline")
        st.markdown(f"**Home country**: {row['home_country']}")
        st.markdown(f"**Avg monthly spend**: ${row['avg_monthly_spend']:.2f}")
        st.markdown(f"**Historical max txn**: ${row['max_historical_txn']:.2f}")
        st.markdown(f"**Typical MCCs**: {row['typical_mcc_list']}")
        ratio = row["amount"] / row["max_historical_txn"] if row["max_historical_txn"] else 0
        st.markdown(f"**Amount vs max**: {ratio:.1f}x")

    # --- Why flagged ---
    st.subheader("Why it was flagged")
    st.markdown(f"**Rules triggered**: `{row['triggered_rules']}`")

    # --- Agent reasoning ---
    st.subheader("Agent reasoning")
    st.info(row["reasoning"])

    st.subheader("Suggested investigation note")
    st.text_area("Note", value=row["investigation_note"], height=100, label_visibility="collapsed")

    # --- Retrieved similar cases ---
    st.subheader("Historical cases the agent considered")
    desc = (
        f"${row['amount']:.2f} at {row['merchant_name']} ({row['mcc_category']}) "
        f"in {row['merchant_country']}, IP {row['ip_country']}, "
        f"customer home {row['home_country']}, rules: {row['triggered_rules']}"
    )
    with st.spinner("Loading similar cases from knowledge base..."):
        similar = find_similar_cases(desc, top_n=3)
    for i, case in enumerate(similar, 1):
        with st.expander(f"Case {i} — similarity {case['similarity']:.2f} — outcome: {case['outcome']}"):
            st.markdown(f"**Pattern tags**: {case['pattern_tags']}")
            st.markdown(f"**Loss amount**: ${case['loss_amount']:.2f}")
            st.markdown(f"**Narrative**: {case['narrative']}")

    # --- Human-in-the-loop action panel ---
    st.markdown("---")
    st.subheader("Your decision")

    if row["analyst_decision"]:
        st.success(
            f"✓ This alert was reviewed at {row['analyst_decided_at']} — "
            f"analyst marked it as **{row['analyst_decision']}**."
        )
        if row["analyst_note"]:
            st.markdown(f"**Note**: {row['analyst_note']}")
    else:
        analyst_note = st.text_input("Optional note", "")
        b1, b2, b3 = st.columns(3)

        with b1:
            if st.button("🚨 Confirm Fraud", use_container_width=True, type="primary"):
                record_analyst_decision(alert_id, "confirmed_fraud", analyst_note)
                # Closed-loop: add a narrative version of this alert to the RAG store
                narrative = (
                    f"Customer at home {row['home_country']} had a ${row['amount']:.2f} "
                    f"{'card-not-present ' if not row['card_present'] else ''}transaction at "
                    f"{row['merchant_name']} ({row['mcc_category']}) in {row['merchant_country']} "
                    f"from IP {row['ip_country']}. "
                    f"Customer's historical max was ${row['max_historical_txn']:.2f} "
                    f"(this transaction was {row['amount']/max(row['max_historical_txn'],1):.1f}x larger). "
                    f"Rules triggered: {row['triggered_rules']}. "
                    f"Analyst confirmed as fraud. {analyst_note}".strip()
                )
                pattern = row["triggered_rules"].split(",")[0] if row["triggered_rules"] else "unknown"
                new_case_id = f"CASE_HUMAN_{alert_id}"
                try:
                    add_case_from_alert(
                        case_id=new_case_id,
                        narrative=narrative,
                        pattern_tags=pattern,
                        outcome="confirmed_fraud",
                        loss_amount=float(row["amount"]),
                    )
                    st.success(f"✓ Fraud confirmed. Added to knowledge base as `{new_case_id}` — "
                               f"next similar alert will use this pattern.")
                except Exception as e:
                    st.warning(f"Decision recorded, but knowledge-base update failed: {e}")
                st.rerun()

        with b2:
            if st.button("✅ Mark False Positive", use_container_width=True):
                record_analyst_decision(alert_id, "false_positive", analyst_note)
                st.success("✓ Cleared as false positive.")
                st.rerun()

        with b3:
            if st.button("⏸️ Needs Deeper Review", use_container_width=True):
                record_analyst_decision(alert_id, "needs_review", analyst_note)
                st.info("⏸ Escalated to senior review.")
                st.rerun()


# =====================================================================
# PAGE 3 — Analytics
# =====================================================================

elif page == "📊 Analytics":
    st.title("Analytics")
    st.caption("Aggregate performance metrics across all alerts the agent has triaged.")

    df = df_all.copy()
    if len(df) == 0:
        st.warning("No data yet. Run `python run_agent.py` first.")
        st.stop()

    # --- KPI strip ---
    decided = df[df["classification"] != "needs_review"]
    fp_alerts = df[df["is_fraud_ground_truth"] == 0]
    fp_cleared = fp_alerts[fp_alerts["classification"] == "false_positive"]
    fraud_alerts = df[df["is_fraud_ground_truth"] == 1]
    fraud_caught = fraud_alerts[fraud_alerts["classification"] == "likely_fraud"]
    fraud_missed = fraud_alerts[fraud_alerts["classification"] == "false_positive"]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total alerts triaged", len(df))
    k2.metric(
        "False-positive reduction",
        f"{len(fp_cleared)/max(len(fp_alerts),1)*100:.0f}%",
        help="% of false-positive alerts the agent correctly auto-cleared",
    )
    k3.metric(
        "Fraud recall",
        f"{len(fraud_caught)/max(len(fraud_alerts),1)*100:.0f}%",
        help="% of real fraud alerts the agent flagged as likely_fraud",
    )
    k4.metric(
        "Fraud missed",
        len(fraud_missed),
        delta=f"{-len(fraud_missed)}" if len(fraud_missed) else "0",
        delta_color="inverse",
        help="Real fraud alerts the agent incorrectly cleared",
    )

    st.markdown("---")

    # --- Two charts side by side ---
    cA, cB = st.columns(2)

    with cA:
        st.subheader("Classification distribution")
        counts = df["classification"].value_counts().reset_index()
        counts.columns = ["classification", "count"]
        fig = px.bar(
            counts, x="classification", y="count",
            color="classification",
            color_discrete_map={
                "likely_fraud": "#c92a2a",
                "needs_review": "#a17008",
                "false_positive": "#2b8a3e",
            },
        )
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)

    with cB:
        st.subheader("Confidence distribution")
        fig2 = px.histogram(
            df, x="confidence", color="classification",
            nbins=20,
            color_discrete_map={
                "likely_fraud": "#c92a2a",
                "needs_review": "#a17008",
                "false_positive": "#2b8a3e",
            },
        )
        fig2.update_layout(height=350, barmode="stack")
        st.plotly_chart(fig2, use_container_width=True)

    # --- Confusion matrix ---
    st.subheader("Confusion matrix (vs ground truth)")
    cm = (
        df.groupby(["is_fraud_ground_truth", "classification"])
        .size().reset_index(name="count")
    )
    cm["ground_truth"] = cm["is_fraud_ground_truth"].map({0: "Not fraud", 1: "Fraud"})
    cm_pivot = cm.pivot(index="ground_truth", columns="classification", values="count").fillna(0)
    fig_cm = px.imshow(
        cm_pivot,
        labels=dict(x="Agent classification", y="Ground truth", color="Count"),
        text_auto=True,
        color_continuous_scale="Blues",
        aspect="auto",
    )
    fig_cm.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_cm, use_container_width=True)

    # --- Per-pattern recall (on real fraud only) ---
    st.subheader("Per-pattern fraud recall")
    fraud_df = df[df["is_fraud_ground_truth"] == 1].copy()
    if len(fraud_df) > 0:
        pattern_stats = (
            fraud_df.groupby("fraud_pattern_ground_truth")
            .agg(
                total=("alert_id", "count"),
                caught=("classification", lambda x: (x == "likely_fraud").sum()),
                missed=("classification", lambda x: (x == "false_positive").sum()),
                to_human=("classification", lambda x: (x == "needs_review").sum()),
            )
            .reset_index()
            .rename(columns={"fraud_pattern_ground_truth": "Pattern"})
        )
        pattern_stats["recall_pct"] = (pattern_stats["caught"] / pattern_stats["total"] * 100).round(1)
        st.dataframe(pattern_stats, use_container_width=True)

    # --- Analyst override rate ---
    reviewed = df[df["analyst_decision"].notna()]
    if len(reviewed) > 0:
        st.subheader("Analyst agreement with agent")
        # Map analyst confirmed_fraud ↔ likely_fraud, false_positive stays as false_positive
        def maps_match(row):
            ad = row["analyst_decision"]
            ac = row["classification"]
            if ad == "confirmed_fraud" and ac == "likely_fraud":
                return "agreed"
            if ad == "false_positive" and ac == "false_positive":
                return "agreed"
            if ad == "needs_review":
                return "escalated"
            return "overrode"
        reviewed = reviewed.copy()
        reviewed["agreement"] = reviewed.apply(maps_match, axis=1)
        agreement_counts = reviewed["agreement"].value_counts().reset_index()
        agreement_counts.columns = ["status", "count"]
        st.dataframe(agreement_counts, use_container_width=True)
        agree_pct = (reviewed["agreement"] == "agreed").sum() / len(reviewed) * 100
        st.metric("Agent-analyst agreement rate", f"{agree_pct:.0f}%")


# =====================================================================
# PAGE 4 — Pattern Explorer
# =====================================================================

elif page == "🔎 Pattern Explorer":
    st.title("Pattern Explorer")
    st.caption(
        "Query the fraud-case knowledge base directly. Type any description, "
        "see the most similar historical cases the agent would retrieve for it."
    )

    query = st.text_area(
        "Describe a scenario",
        value="Customer had multiple small online charges in 5 minutes from foreign IPs",
        height=80,
    )
    top_n = st.slider("Number of cases to retrieve", 1, 10, 5)

    if st.button("🔍 Search knowledge base"):
        with st.spinner("Searching..."):
            results = find_similar_cases(query, top_n=top_n)

        st.markdown(f"**{len(results)} results** for your query.")
        for i, case in enumerate(results, 1):
            with st.expander(
                f"#{i} — {case['case_id']} — similarity {case['similarity']:.3f} — "
                f"outcome: **{case['outcome']}**"
            ):
                st.markdown(f"**Pattern tags**: {case['pattern_tags']}")
                st.markdown(f"**Loss amount**: ${case['loss_amount']:.2f}")
                st.markdown(f"**Narrative**:")
                st.write(case["narrative"])

    st.markdown("---")
    st.markdown("**Browse all cases in the knowledge base**")
    conn = sqlite3.connect(DB_PATH)
    cases_df = pd.read_sql(
        "SELECT case_id, pattern_tags, outcome, loss_amount, substr(narrative, 1, 100) || '...' AS preview "
        "FROM fraud_cases ORDER BY case_id",
        conn,
    )
    conn.close()
    st.dataframe(cases_df, use_container_width=True, height=400)