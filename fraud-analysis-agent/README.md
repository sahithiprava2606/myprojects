# Vectorless RAG with PageIndex

A minimal **Retrieval-Augmented Generation** pipeline that answers questions over a PDF **without using a vector database**. Instead of chunking + embedding + similarity search, this approach uses [PageIndex](https://pageindex.ai) to build a hierarchical tree of the document (like a smart table of contents) and lets an LLM navigate that tree to find the right sections.

## How it works

```
PDF ──► PageIndex tree ──► LLM tree search ──► Retrieve nodes ──► Generate cited answer
```

1. **Upload & index** — submit the PDF to PageIndex, which builds a hierarchical tree of titles, summaries, and section text.
2. **Tree search** — pass a compressed view of the tree to an LLM and ask it which `node_id`s are most relevant to the query.
3. **Retrieve** — walk the tree and pull the full text of the selected nodes.
4. **Answer** — feed the retrieved sections back to the LLM, which produces an answer with section + page citations.

## Why "vectorless"?

No embeddings, no vector store, no chunk-size tuning. Retrieval reasoning is delegated to the LLM, using the document's natural structure rather than vector similarity. Trade-off: more LLM calls per query, but no embedding infrastructure and better explainability (you can see *why* a section was picked).

## Requirements

- Python 3.12+
- A [PageIndex](https://pageindex.ai) API key
- An [OpenRouter](https://openrouter.ai) API key (or swap in OpenAI/Anthropic directly)

### Install dependencies

```bash
pip install pageindex openai python-dotenv
```

## Setup

Create a `.env` file in the project root:

```env
PAGEINDEX_API_KEY=your_pageindex_key_here
OPENROUTER_API_KEY=your_openrouter_key_here
```

> **Security note:** the notebook currently hardcodes both API keys in cells 1 and 2. Move them into `.env` and load them with `os.getenv(...)` before sharing the notebook anywhere. Rotate any keys that were committed.

## Usage

Open `vecorless_rag.ipynb` and run cells top-to-bottom:

1. **Cells 1–2** — load API keys and initialize the PageIndex + OpenAI clients.
2. **Cell 3** — set `PDF_PATH` to your document and upload it.
3. **Cell 4** — poll until PageIndex finishes building the tree (~30–90s for a 50-page PDF).
4. **Cells 5–6** — inspect the tree structure.
5. **Cells 7–9** — defines `llm_tree_search`, `find_nodes_by_ids`, `generate_answer`.
6. **Cell 10** — the full pipeline wrapped in `vectorless_rag(query, tree)`.
7. **Cell 11** — run a sample query.

### Example

```python
answer = vectorless_rag(
    query="Give me a summary of the key points in the document?",
    tree=pageindex_tree
)
```

Output includes the LLM's reasoning, the selected node IDs, the matched section titles, and a final answer with citations like `(Definition of PFP, Page 2)`.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_NAME` | `google/gemini-2.0-flash-001` | LLM used for tree search and answer generation |
| `PDF_PATH` | *(set in cell 3)* | Local path to the PDF you want to query |

Swap `MODEL_NAME` for any model your OpenRouter account has access to (e.g. `anthropic/claude-sonnet-4`, `openai/gpt-4o`).

## Caching the tree

The notebook does **not** save the PageIndex tree to disk — it lives only in the `pageindex_tree` variable for the kernel's lifetime. To avoid re-fetching every session:

```python
import json

# Save once
with open("pageindex_tree.json", "w") as f:
    json.dump(pageindex_tree, f, indent=2)

# Reload later
with open("pageindex_tree.json") as f:
    pageindex_tree = json.load(f)
```

The canonical copy also lives on PageIndex's servers, retrievable via `pi_client.get_document(doc_id)`.

## File structure

```
.
├── vecorless_rag.ipynb     # main notebook
├── sample_document.pdf     # your input PDF
├── pageindex_tree.json     # (optional) cached tree
├── .env                    # API keys
└── README.md
```

## Limitations

- Each query makes **two LLM calls** (tree search + answer generation), so latency and cost scale with traffic.
- The compressed tree must fit in the LLM's context window — works well for documents up to a few hundred pages, but very large corpora will need batching or a hybrid approach.
- Quality of retrieval depends on how well-structured the source PDF is. Documents with meaningful section headings work best.

## License

Add your preferred license here.
