"""
RAG (Retrieval-Augmented Generation) module for the LLM pipeline.
"""

from .router import QueryRouter, QueryIntent, RoutingResult
from .retriever_v2 import (
    BaseRetriever, 
    VectorRetriever, 
    ConnectorRetriever, 
    HybridRetriever,
    RetrievalResult,
    create_retriever
)
from .ingest import DocumentChunk, load_document, load_directory
from .vector_store import VectorStore

# For backward compatibility
Retriever = VectorRetriever

__all__ = [
    # Core components
    'BaseRetriever',
    'VectorRetriever',
    'ConnectorRetriever',
    'HybridRetriever',
    'RetrievalResult',
    'create_retriever',
    'VectorStore',
    'DocumentChunk',
    'load_document',
    'load_directory',
    
    # Router
    'QueryRouter', 
    'QueryIntent', 
    'RoutingResult',
    
    # For backward compatibility
    'Retriever'
]
