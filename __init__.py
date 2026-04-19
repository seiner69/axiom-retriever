"""axiom-retriever: A modular retrieval library for RAG.

Provides retrieval strategies for finding relevant chunks from a vector store.

Example:
    >>> from axiom_retriever.strategies import SimilarityRetriever
    >>>
    >>> retriever = SimilarityRetriever(embedder=embedder, vector_store=store)
    >>> result = retriever.retrieve("What is the revenue?", top_k=5)
    >>> print(f"Retrieved {len(result.chunks)} chunks")
"""

from axiom_retriever.core import (
    BaseRetriever,
    RetrievedChunk,
    RetrievalMode,
    RetrievalResult,
)
from axiom_retriever.strategies import MMRRetriever, SimilarityRetriever

__all__ = [
    # Core
    "BaseRetriever",
    "RetrievedChunk",
    "RetrievalMode",
    "RetrievalResult",
    # Strategies
    "SimilarityRetriever",
    "MMRRetriever",
]

__version__ = "0.1.0"
