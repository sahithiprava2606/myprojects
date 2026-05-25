"""
Synthetic transaction dataset generator for the Transaction Anomaly Triage Agent.

Produces a SQLite database (fraud_data.db) with five tables:
  - customers, merchants, devices, transactions, fraud_cases

Designed so:
  1. Most transactions follow each customer's "spending personality" (baseline).
  2. ~1.5% of transactions are fraud, embedded as one of five named patterns.
  3. The patterns are detectable by simple rules (velocity, geo, amount, MCC, CNP).

Run:
    pip install faker pandas numpy
    python generate_dataset.py
"""

import random
import sqlite3
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

# Reproducibility — same seed gives same dataset
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
fake = Faker()
Faker.seed(SEED)

# ---------- Configuration ----------
NUM_CUSTOMERS = 5_000
NUM_MERCHANTS = 500
NUM_TRANSACTIONS = 100_000
NUM_FRAUD_CASES = 80
FRAUD_RATE = 0.015  # 1.5% of transactions are fraud
DAYS_OF_HISTORY = 90
END_DATE = datetime(2026, 5, 21)
START_DATE = END_DATE - timedelta(days=DAYS_OF_HISTORY)

DB_PATH = Path("fraud_data.db")

# ---------- Reference data ----------
# Merchant Category Codes (MCC) grouped into spending personalities.
# Each customer will prefer 2-4 categories from these groups.
MCC_CATEGORIES = {
    "grocery":       {"risk": "low",    "typical_amount": (15, 200)},
    "gas":           {"risk": "low",    "typical_amount": (20, 80)},
    "restaurant":    {"risk": "low",    "typical_amount": (10, 150)},
    "retail":        {"risk": "low",    "typical_amount": (20, 500)},
    "online_retail": {"risk": "medium", "typical_amount": (15, 800)},
    "travel":        {"risk": "medium", "typical_amount": (100, 2000)},
    "electronics":   {"risk": "medium", "typical_amount": (50, 3000)},
    "jewelry":       {"risk": "high",   "typical_amount": (200, 5000)},
    "crypto":        {"risk": "high",   "typical_amount": (50, 2000)},
    "gambling":      {"risk": "high",   "typical_amount": (20, 1000)},
    "money_transfer":{"risk": "high",   "typical_amount": (50, 3000)},
}

# Countries: weighted so most customers are in 5 "home" countries
HOME_COUNTRIES = ["US", "UK", "CA", "AU", "IN"]
FOREIGN_COUNTRIES = ["MX", "BR", "RU", "NG", "CN", "TH", "DE", "FR", "JP", "AE"]


# ---------- Generators ----------

def generate_customers(n):
    """Each customer has a spending personality the agent will learn."""
    rows = []
    for _ in range(n):
        home_country = random.choices(HOME_COUNTRIES, weights=[5, 2, 2, 1, 2])[0]
        # 2-4 typical MCC categories drawn from low-risk-leaning distribution
        low_risk_mccs = ["grocery", "gas", "restaurant", "retail", "online_retail"]
        other_mccs = ["travel", "electronics"]
        num_typical = random.randint(2, 4)
        typical_mccs = random.sample(low_risk_mccs, k=min(num_typical, len(low_risk_mccs)))
        if random.random() < 0.3:
            typical_mccs.append(random.choice(other_mccs))

        avg_spend = round(np.random.lognormal(mean=6.2, sigma=0.6), 2)  # ~$500/mo median
        max_txn = round(avg_spend * np.random.uniform(2, 6), 2)

        rows.append({
            "customer_id": f"CUST_{uuid.uuid4().hex[:10].upper()}",
            "name": fake.name(),
            "email": fake.email(),
            "home_country": home_country,
            "home_city": fake.city(),
            "account_open_date": fake.date_between(start_date="-5y", end_date="-30d"),
            "avg_monthly_spend": avg_spend,
            "max_historical_txn": max_txn,
            "typical_mcc_list": ",".join(typical_mccs),
        })
    return pd.DataFrame(rows)


def generate_merchants(n):
    rows = []
    for _ in range(n):
        mcc = random.choice(list(MCC_CATEGORIES.keys()))
        risk = MCC_CATEGORIES[mcc]["risk"]
        # High-risk merchants more likely to be in foreign countries
        if risk == "high":
            country = random.choices(HOME_COUNTRIES + FOREIGN_COUNTRIES, weights=[1]*5 + [3]*10)[0]
        else:
            country = random.choices(HOME_COUNTRIES + FOREIGN_COUNTRIES, weights=[6]*5 + [1]*10)[0]

        rows.append({
            "merchant_id": f"MERCH_{uuid.uuid4().hex[:8].upper()}",
            "merchant_name": fake.company(),
            "mcc_category": mcc,
            "country": country,
            "risk_tier": risk,
        })
    return pd.DataFrame(rows)


def generate_devices(customers_df):
    """Most customers have 1-2 devices; some have 3."""
    rows = []
    for cust_id in customers_df["customer_id"]:
        num_devices = random.choices([1, 2, 3], weights=[60, 30, 10])[0]
        for _ in range(num_devices):
            rows.append({
                "device_id": f"DEV_{uuid.uuid4().hex[:10].upper()}",
                "customer_id": cust_id,
                "device_type": random.choice(["mobile_ios", "mobile_android", "desktop", "tablet"]),
                "first_seen": fake.date_between(start_date="-2y", end_date="today"),
            })
    return pd.DataFrame(rows)


def generate_transactions(customers_df, merchants_df, devices_df, n):
    """
    Generate normal transactions following each customer's personality,
    then inject fraud transactions in five distinct patterns.
    """
    n_fraud = int(n * FRAUD_RATE)
    n_normal = n - n_fraud

    # Index for fast lookups
    cust_idx = customers_df.set_index("customer_id")
    merch_by_mcc = merchants_df.groupby("mcc_category")["merchant_id"].apply(list).to_dict()
    devices_by_cust = devices_df.groupby("customer_id")["device_id"].apply(list).to_dict()
    merch_idx = merchants_df.set_index("merchant_id")

    rows = []

    # ---- Normal transactions ----
    print(f"  generating {n_normal:,} normal transactions...")
    customer_ids = customers_df["customer_id"].tolist()
    for i in range(n_normal):
        cust_id = random.choice(customer_ids)
        cust = cust_idx.loc[cust_id]
        typical_mccs = cust["typical_mcc_list"].split(",")
        mcc = random.choice(typical_mccs)
        if mcc not in merch_by_mcc:
            mcc = random.choice(list(merch_by_mcc.keys()))
        merchant_id = random.choice(merch_by_mcc[mcc])
        merch = merch_idx.loc[merchant_id]

        # Amount drawn from category's typical range, biased toward customer's avg
        amt_low, amt_high = MCC_CATEGORIES[mcc]["typical_amount"]
        amount = round(np.random.uniform(amt_low, min(amt_high, cust["max_historical_txn"])), 2)

        device_id = random.choice(devices_by_cust[cust_id])
        # Normal transactions: ip_country matches home_country 95% of the time
        ip_country = cust["home_country"] if random.random() < 0.95 else random.choice(FOREIGN_COUNTRIES)
        ts = fake.date_time_between(start_date=START_DATE, end_date=END_DATE)

        rows.append({
            "transaction_id": f"TXN_{uuid.uuid4().hex[:12].upper()}",
            "customer_id": cust_id,
            "merchant_id": merchant_id,
            "device_id": device_id,
            "timestamp": ts,
            "amount": amount,
            "currency": "USD",
            "card_present": random.random() < 0.4,
            "ip_country": ip_country,
            "is_fraud": False,
            "fraud_pattern": None,
        })
        if (i+1) % 20000 == 0:
            print(f"    {i+1:,}/{n_normal:,}")

    # ---- Fraud transactions ----
    print(f"  injecting {n_fraud:,} fraud transactions across 5 patterns...")
    fraud_customer_ids = random.sample(customer_ids, k=min(n_fraud, len(customer_ids)))
    patterns = ["velocity", "geo_impossible", "amount_anomaly", "mcc_anomaly", "cnp_foreign"]

    for cust_id in fraud_customer_ids:
        cust = cust_idx.loc[cust_id]
        pattern = random.choice(patterns)
        device_id = random.choice(devices_by_cust[cust_id])
        base_ts = fake.date_time_between(start_date=START_DATE, end_date=END_DATE)

        if pattern == "velocity":
            # 5-7 small transactions within 10 minutes — card testing
            count = random.randint(5, 7)
            for j in range(count):
                merchant_id = random.choice(merch_by_mcc.get("online_retail", merchants_df["merchant_id"].tolist()))
                rows.append({
                    "transaction_id": f"TXN_{uuid.uuid4().hex[:12].upper()}",
                    "customer_id": cust_id,
                    "merchant_id": merchant_id,
                    "device_id": device_id,
                    "timestamp": base_ts + timedelta(seconds=j*90),
                    "amount": round(random.uniform(1, 25), 2),
                    "currency": "USD",
                    "card_present": False,
                    "ip_country": random.choice(FOREIGN_COUNTRIES),
                    "is_fraud": True,
                    "fraud_pattern": "velocity",
                })

        elif pattern == "geo_impossible":
            # Two transactions, far apart, within 30 mins
            local_merch = random.choice(merch_by_mcc.get("retail", merchants_df["merchant_id"].tolist()))
            foreign_merchants = merchants_df[merchants_df["country"].isin(FOREIGN_COUNTRIES)]["merchant_id"].tolist()
            foreign_merch = random.choice(foreign_merchants)
            rows.append({
                "transaction_id": f"TXN_{uuid.uuid4().hex[:12].upper()}",
                "customer_id": cust_id, "merchant_id": local_merch, "device_id": device_id,
                "timestamp": base_ts, "amount": round(random.uniform(20, 200), 2),
                "currency": "USD", "card_present": True, "ip_country": cust["home_country"],
                "is_fraud": False, "fraud_pattern": None,  # the *local* txn is genuine
            })
            rows.append({
                "transaction_id": f"TXN_{uuid.uuid4().hex[:12].upper()}",
                "customer_id": cust_id, "merchant_id": foreign_merch, "device_id": device_id,
                "timestamp": base_ts + timedelta(minutes=random.randint(15, 30)),
                "amount": round(random.uniform(100, 2000), 2),
                "currency": "USD", "card_present": False,
                "ip_country": random.choice(FOREIGN_COUNTRIES),
                "is_fraud": True, "fraud_pattern": "geo_impossible",
            })

        elif pattern == "amount_anomaly":
            # One transaction 8-15x the customer's historical max
            multiplier = random.uniform(8, 15)
            amount = round(cust["max_historical_txn"] * multiplier, 2)
            merchant_id = random.choice(merch_by_mcc.get("electronics", merchants_df["merchant_id"].tolist()))
            rows.append({
                "transaction_id": f"TXN_{uuid.uuid4().hex[:12].upper()}",
                "customer_id": cust_id, "merchant_id": merchant_id, "device_id": device_id,
                "timestamp": base_ts, "amount": amount,
                "currency": "USD", "card_present": False,
                "ip_country": cust["home_country"],
                "is_fraud": True, "fraud_pattern": "amount_anomaly",
            })

        elif pattern == "mcc_anomaly":
            # Transaction in a high-risk category the customer never uses
            unusual_mcc = random.choice(["crypto", "gambling", "money_transfer"])
            if unusual_mcc in cust["typical_mcc_list"]:
                unusual_mcc = "jewelry"
            merchant_id = random.choice(merch_by_mcc.get(unusual_mcc, merchants_df["merchant_id"].tolist()))
            rows.append({
                "transaction_id": f"TXN_{uuid.uuid4().hex[:12].upper()}",
                "customer_id": cust_id, "merchant_id": merchant_id, "device_id": device_id,
                "timestamp": base_ts, "amount": round(random.uniform(300, 2500), 2),
                "currency": "USD", "card_present": False,
                "ip_country": random.choice(FOREIGN_COUNTRIES),
                "is_fraud": True, "fraud_pattern": "mcc_anomaly",
            })

        elif pattern == "cnp_foreign":
            # Card-not-present in a country the customer has never transacted from
            foreign_merchants = merchants_df[merchants_df["country"].isin(FOREIGN_COUNTRIES)]["merchant_id"].tolist()
            merchant_id = random.choice(foreign_merchants)
            rows.append({
                "transaction_id": f"TXN_{uuid.uuid4().hex[:12].upper()}",
                "customer_id": cust_id, "merchant_id": merchant_id, "device_id": device_id,
                "timestamp": base_ts, "amount": round(random.uniform(200, 1500), 2),
                "currency": "USD", "card_present": False,
                "ip_country": random.choice(FOREIGN_COUNTRIES),
                "is_fraud": True, "fraud_pattern": "cnp_foreign",
            })

    df = pd.DataFrame(rows)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def generate_fraud_cases(n):
    """
    Closed historical fraud cases as narratives.
    These become the corpus for the RAG vector store.
    """
    templates = [
        ("velocity", "confirmed_fraud",
         "Customer reported card lost after noticing {count} small online retail charges of $1-$25 each within {minutes} minutes, all from IP addresses in {country}. Pattern matched known card-testing behavior where fraudsters validate stolen card numbers with low-value transactions before attempting a larger purchase. No prior travel to {country} on file. Card was blocked and reissued."),
        ("geo_impossible", "confirmed_fraud",
         "Genuine in-person purchase at {merchant} in {home_country} at {time1}, followed by an online purchase of ${amount} from {country} at {time2} — physically impossible travel window. Customer confirmed the foreign transaction was unauthorized. Likely card-number compromise via prior skimming or data breach. Loss recovered through chargeback."),
        ("amount_anomaly", "confirmed_fraud",
         "Customer's 90-day spending baseline was approximately ${baseline}/month with no single transaction over ${max}. Flagged charge of ${amount} at an electronics retailer was {multiplier}x the historical maximum. Customer had not authorized the purchase. Fraudster used stolen credentials to order high-resale-value goods for shipment to a forwarding address."),
        ("mcc_anomaly", "confirmed_fraud",
         "Customer's transaction history showed exclusive use of grocery, gas, and restaurant categories for 18 months. Sudden ${amount} charge at a cryptocurrency exchange from a foreign IP triggered the alert. Customer confirmed they have never used crypto services. Account credentials had been phished via a fake banking SMS two days prior."),
        ("cnp_foreign", "confirmed_fraud",
         "Card-not-present transaction of ${amount} at an online merchant based in {country}, originating from an IP in the same country. Customer's home country is {home_country} and there is no record of international transactions in 24 months. Customer confirmed unauthorized use. The card number had appeared in a recent dark web data dump."),
        ("velocity", "false_positive",
         "Customer made {count} legitimate small purchases at a single online merchant within {minutes} minutes while completing a multi-step booking. All charges were from the same merchant ID and same device. Investigation confirmed it was a single split-payment authorization flow. No fraud, customer not contacted, alert auto-closed after merchant verification."),
        ("geo_impossible", "false_positive",
         "Customer was traveling and used both their physical card at a hotel in {country} and their virtual card via mobile app while on the same trip. Geographic distance between the two charges appeared suspicious but device and IP records confirmed both came from the customer. False positive, no action needed."),
        ("amount_anomaly", "false_positive",
         "Charge of ${amount} flagged for being {multiplier}x the customer's typical spending. Customer confirmed they had purchased a new appliance and that this was a planned, one-off expense. Receipt was provided. Baseline was updated to reflect the new spending event. No fraud."),
        ("mcc_anomaly", "customer_error",
         "Transaction at a gambling merchant flagged because the customer had no history in that category. Customer confirmed they had signed up for a new fantasy sports platform and had forgotten about the deposit. No fraud — customer reminded to monitor authorized recurring charges."),
        ("cnp_foreign", "false_positive",
         "Online subscription renewal for a service headquartered in {country}, billed in USD to the customer in {home_country}. Customer had this subscription for 14 months but it had previously been billed by a domestic processor. Vendor recently moved billing to their home country, triggering the geo rule. No fraud."),
    ]

    rows = []
    for i in range(n):
        pattern, outcome, template = random.choice(templates)
        narrative = template.format(
            count=random.randint(5, 9),
            minutes=random.randint(3, 12),
            country=random.choice(FOREIGN_COUNTRIES),
            home_country=random.choice(HOME_COUNTRIES),
            merchant=fake.company(),
            time1=f"{random.randint(8,18)}:{random.randint(10,59)}",
            time2=f"{random.randint(8,18)}:{random.randint(10,59)}",
            amount=round(random.uniform(200, 5000), 2),
            baseline=round(random.uniform(300, 1500), 2),
            max=round(random.uniform(200, 800), 2),
            multiplier=random.randint(6, 18),
        )
        rows.append({
            "case_id": f"CASE_{2024-random.randint(0,3)}_{i:04d}",
            "narrative": narrative,
            "pattern_tags": pattern,
            "outcome": outcome,
            "loss_amount": round(random.uniform(0, 8000), 2) if outcome == "confirmed_fraud" else 0.0,
        })
    return pd.DataFrame(rows)


# ---------- Main ----------

def main():
    print("Generating customers...")
    customers = generate_customers(NUM_CUSTOMERS)

    print("Generating merchants...")
    merchants = generate_merchants(NUM_MERCHANTS)

    print("Generating devices...")
    devices = generate_devices(customers)

    print("Generating transactions...")
    transactions = generate_transactions(customers, merchants, devices, NUM_TRANSACTIONS)

    print("Generating fraud case history...")
    fraud_cases = generate_fraud_cases(NUM_FRAUD_CASES)

    # Write to SQLite
    print(f"\nWriting to {DB_PATH}...")
    if DB_PATH.exists():
        DB_PATH.unlink()
    conn = sqlite3.connect(DB_PATH)
    customers.to_sql("customers", conn, index=False)
    merchants.to_sql("merchants", conn, index=False)
    devices.to_sql("devices", conn, index=False)
    transactions.to_sql("transactions", conn, index=False)
    fraud_cases.to_sql("fraud_cases", conn, index=False)

    # Indexes for the rules engine — without these, customer-history queries crawl
    conn.executescript("""
        CREATE INDEX idx_txn_customer ON transactions(customer_id);
        CREATE INDEX idx_txn_timestamp ON transactions(timestamp);
        CREATE INDEX idx_txn_customer_time ON transactions(customer_id, timestamp);
        CREATE INDEX idx_devices_customer ON devices(customer_id);
    """)
    conn.commit()

    # Summary stats
    print("\n=== Dataset summary ===")
    print(f"Customers:    {len(customers):>8,}")
    print(f"Merchants:    {len(merchants):>8,}")
    print(f"Devices:      {len(devices):>8,}")
    print(f"Transactions: {len(transactions):>8,}")
    print(f"  fraud:      {transactions['is_fraud'].sum():>8,}  ({transactions['is_fraud'].mean()*100:.2f}%)")
    print(f"  by pattern:")
    for pat, ct in transactions[transactions["is_fraud"]]["fraud_pattern"].value_counts().items():
        print(f"    {pat:<18} {ct:>6,}")
    print(f"Fraud cases:  {len(fraud_cases):>8,}")
    print(f"\nDatabase ready at: {DB_PATH.resolve()}")
    conn.close()


if __name__ == "__main__":
    main()