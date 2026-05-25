# 🛡️ Transaction Anomaly Triage Agent

An agentic AI system that triages credit-card fraud alerts. For each flagged transaction, it pulls customer context via SQL, retrieves similar past cases via RAG, classifies the alert with reasoning, and drafts an investigation note — so analysts only review the genuinely risky ones.

![demo](docs/demo.gif)

## Results

Evaluated on 50 balanced alerts (25 fraud, 25 false positive) using Claude Sonnet 4:

- **92% fraud recall** — 23 of 25 real fraud cases caught
- **76% false-positive reduction** — 19 of 25 noise alerts auto-cleared
- **84% accuracy** on alerts the agent committed to (vs `needs_review`)
- **100% recall** on velocity, geo-impossible, mcc-anomaly, and amount-anomaly fraud patterns

## Stack

LangGraph · ChromaDB · SQLite · OpenRouter (Claude Sonnet 4) · sentence-transformers · Streamlit · Plotly

## Architecture

```
100k transactions → Rules engine → 5,800 alerts → LangGraph agent → Streamlit dashboard
                    (5 patterns)                   (SQL + RAG + LLM)   (human-in-the-loop)
```

The agent runs each alert through 6 nodes: fetch SQL context → compose description → retrieve similar cases → novelty gate → LLM classify → confidence gate. Confirmed fraud from the dashboard is auto-added to the RAG knowledge base, closing the learning loop.

## Quick start

```bash
pip install -r requirements.txt
export OPENROUTER_API_KEY="sk-or-..."

python generate_dataset.py    # 100k synthetic transactions
python rules_engine.py        # produces alerts
python vector_store.py        # embeds 80 historical cases
python run_agent.py           # triage a sample of 50 alerts
streamlit run dashboard.py    # launch the UI
```

## What this project demonstrates

- **Agentic AI** with LangGraph: stateful workflow, conditional routing, defensive gates
- **RAG** with ChromaDB: semantic retrieval, novelty thresholds, closed-loop knowledge updates
- **SQL**: time-windowed customer context queries on 100k+ rows
- **LLM engineering**: structured output with Pydantic, model-agnostic via OpenRouter, prompt iteration with paired metrics
- **Production thinking**: human-in-the-loop, confidence calibration, auditable reasoning with case citations

## The honest story

The agent took 8 iterations to reach production-quality numbers. The biggest lesson came from a Python-vs-SQLite timestamp format bug that silently made my burst-window SQL return zero rows — I'd been tuning the prompt for 4 iterations against data the agent couldn't see. *When prompts won't converge, suspect the data plumbing first.*
