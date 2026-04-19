# magic-retriever

Modular retrieval library for RAG applications.

## Features

| Strategy | Class | Description |
|----------|-------|-------------|
| Similarity | `SimilarityRetriever` | Embed query + cosine similarity search |
| MMR | `MMRRetriever` | Maximal Marginal Relevance for diversity |

## Installation

```bash
pip install sentence-transformers
```

## Quick Start

```python
from magic_retriever.strategies import SimilarityRetriever
from magic_embedder.strategies import SentenceTransformerEmbedder
from magic_vectorstore import ChromaVectorStore

embedder = SentenceTransformerEmbedder()
store = ChromaVectorStore(collection_name="my_collection")
retriever = SimilarityRetriever(embedder=embedder, vector_store=store)

result = retriever.retrieve("What is the revenue?", top_k=5)
print(f"Retrieved {len(result.chunks)} chunks")
```

## MMR Retrieval

```python
from magic_retriever.strategies import MMRRetriever

mmr = MMRRetriever(
    embedder=embedder,
    vector_store=store,
    top_k=5,
    fetch_k=20,
    lambda_mult=0.5,  # 0=max diversity, 1=max relevance
)
result = mmr.retrieve("What is the revenue?", top_k=5)
```

## Module Structure

```
magic_retriever/
    __init__.py
    run.py
    core/           # RetrievedChunk, RetrievalResult, BaseRetriever
    strategies/
        similarity_retriever.py
        mmr_retriever.py
    utils/
```
