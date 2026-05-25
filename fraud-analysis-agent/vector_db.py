"""
Phase 3 — Vector store of fraud patterns.

Reads narratives from fraud_data.db (fraud_cases table), embeds them,
and stores them in ChromaDB. Exposes find_similar_cases() for the agent.

Run once to build the store:
    pip install chromadb sentence-transformers
    python vector_store.py

After the first run, the store persists in ./chroma_db/.
Importing this module from the agent will reuse the existing store.
"""

import sqlite3
from pathlib import Path
from typing import List, Dict

import chromadb
from chromadb.utils import embedding_functions

# ---------- Configuration ----------
DB_PATH = Path("fraud_data.db")
CHROMA_DIR = Path("chroma_db")
COLLECTION_NAME = "fraud_cases"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 80MB, runs on CPU, free

# Novelty threshold — alerts below this go to needs_review without LLM call.
# Tune empirically in Phase 6; 0.55 is a reasonable starting point.
NOVELTY_THRESHOLD = 0.55


# ---------- Build the store ----------

def build_store(rebuild: bool = False):
    """
    Load narratives from SQLite, embed them, write to ChromaDB.
    Call with rebuild=True to wipe and recreate (e.g. after adding new cases).
    """
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    # If rebuilding, drop existing collection
    existing = [c.name for c in client.list_collections()]
    if rebuild and COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
        existing.remove(COLLECTION_NAME)

    # Skip rebuild if already populated
    if COLLECTION_NAME in existing:
        collection = client.get_collection(name=COLLECTION_NAME, embedding_function=embedder)
        if collection.count() > 0:
            print(f"Collection '{COLLECTION_NAME}' already populated with {collection.count()} cases.")
            print("Pass rebuild=True to recreate.")
            return collection

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedder,
        metadata={"hnsw:space": "cosine"},  # cosine similarity for distances
    )

    # Load narratives from SQLite
    print(f"Loading fraud cases from {DB_PATH}...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT case_id, narrative, pattern_tags, outcome, loss_amount FROM fraud_cases")
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        raise RuntimeError("No rows in fraud_cases table. Run generate_dataset.py first.")

    print(f"Embedding {len(rows)} narratives with {EMBEDDING_MODEL}...")
    ids = [r[0] for r in rows]
    documents = [r[1] for r in rows]
    metadatas = [
        {"pattern_tags": r[2], "outcome": r[3], "loss_amount": float(r[4])}
        for r in rows
    ]

    # ChromaDB embeds in batches automatically
    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    print(f"Stored {collection.count()} cases in ChromaDB at {CHROMA_DIR.resolve()}")
    return collection


# ---------- Query interface (used by the agent in Phase 4) ----------

def find_similar_cases(description: str, top_n: int = 3) -> List[Dict]:
    """
    Return the top_n cases most similar to the given description.

    Each returned dict has:
        case_id, narrative, pattern_tags, outcome, loss_amount, similarity

    'similarity' is cosine similarity (0 to 1). Higher means more similar.
    Use the similarity score to decide whether retrievals are strong enough
    to trust — see NOVELTY_THRESHOLD.
    """
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    collection = client.get_collection(name=COLLECTION_NAME, embedding_function=embedder)

    results = collection.query(
        query_texts=[description],
        n_results=top_n,
    )

    matches = []
    for case_id, doc, meta, distance in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        # ChromaDB returns cosine DISTANCE (0 = identical, 2 = opposite).
        # Convert to similarity (1 = identical, -1 = opposite).
        similarity = 1.0 - distance
        matches.append({
            "case_id": case_id,
            "narrative": doc,
            "pattern_tags": meta["pattern_tags"],
            "outcome": meta["outcome"],
            "loss_amount": meta["loss_amount"],
            "similarity": round(similarity, 4),
        })
    return matches


def is_novel(matches: List[Dict]) -> bool:
    """
    True if the strongest retrieval is below the novelty threshold,
    meaning the agent has no good precedent and should route to human.
    """
    if not matches:
        return True
    return max(m["similarity"] for m in matches) < NOVELTY_THRESHOLD


# ---------- Add new case from analyst confirmation (closed loop) ----------

def add_case_from_alert(
    case_id: str,
    narrative: str,
    pattern_tags: str,
    outcome: str = "confirmed_fraud",
    loss_amount: float = 0.0,
):
    """
    Used by the dashboard when an analyst confirms a fraud.
    Writes the case to SQLite AND embeds it into ChromaDB immediately,
    so the next alert benefits from this knowledge.
    """
    # 1. Persist to SQLite (so it survives a vector-store rebuild)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO fraud_cases (case_id, narrative, pattern_tags, outcome, loss_amount) "
        "VALUES (?, ?, ?, ?, ?)",
        (case_id, narrative, pattern_tags, outcome, loss_amount),
    )
    conn.commit()
    conn.close()

    # 2. Add to ChromaDB (embedded automatically)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    collection = client.get_collection(name=COLLECTION_NAME, embedding_function=embedder)
    collection.add(
        ids=[case_id],
        documents=[narrative],
        metadatas=[{
            "pattern_tags": pattern_tags,
            "outcome": outcome,
            "loss_amount": float(loss_amount),
        }],
    )
    print(f"Added case {case_id} to knowledge base. Total cases: {collection.count()}")


# ---------- Sanity test ----------

def sanity_test():
    """Search with a fake alert description, print the matches."""
    print("\n=== Sanity test ===")
    test_descriptions = [
        # Should match velocity / card-testing cases
        "Customer had 6 small online charges of $5-$20 each in 4 minutes from foreign IPs",
        # Should match amount-anomaly cases
        "Customer's typical spend is $400/month, sudden $3,500 electronics purchase from new device",
        # Should match cnp-foreign cases
        "Online subscription renewal from a country the customer has never transacted with",
    ]

    for desc in test_descriptions:
        print(f"\nQuery: {desc[:80]}...")
        matches = find_similar_cases(desc, top_n=3)
        novel = is_novel(matches)
        print(f"  Novel pattern (no strong match)? {novel}")
        for i, m in enumerate(matches, 1):
            print(f"  {i}. similarity={m['similarity']:.3f} | "
                  f"pattern={m['pattern_tags']:<16} | outcome={m['outcome']}")
            # Show first 120 chars of narrative
            print(f"     {m['narrative'][:120]}...")


# ---------- Main ----------

if __name__ == "__main__":
    build_store()
    sanity_test()