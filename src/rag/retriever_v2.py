"""
Enhanced Retriever module for the RAG pipeline.
Supports multiple data sources including documents and external APIs.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Type, AsyncGenerator
from pathlib import Path
import logging
import json
import asyncio
from dataclasses import dataclass

from .ingest import DocumentChunk, load_document, load_directory
from .vector_store import VectorStore
from .connectors.base import BaseConnector, DocumentConnector, APIConnector
from .connectors.odoo_connector import OdooConnector

logger = logging.getLogger(__name__)

@dataclass
class RetrievalResult:
    """Result of a retrieval operation."""
    content: str
    metadata: Dict[str, Any]
    source_type: str
    score: float = 1.0
    chunk_id: Optional[str] = None

class BaseRetriever(ABC):
    """Base class for all retrievers."""
    
    @abstractmethod
    async def retrieve(
        self, 
        query: str,
        query_embedding: List[float],
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        **kwargs
    ) -> List[RetrievalResult]:
        """Retrieve relevant information for a query."""
        pass
    
    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """Get schema information for the retriever's data source."""
        pass

class VectorRetriever(BaseRetriever):
    """Retriever for vector-based document search."""
    
    def __init__(
        self, 
        vector_store: Optional[VectorStore] = None,
        index_path: Optional[Union[str, Path]] = None,
        embedding_dim: int = 768
    ):
        """
        Initialize the vector retriever.
        
        Args:
            vector_store: Pre-initialized vector store (optional)
            index_path: Path to load/save the vector store index
            embedding_dim: Dimension of the embeddings
        """
        self.vector_store = vector_store or VectorStore(
            index_path=index_path,
            dimension=embedding_dim
        )
        self.embedding_dim = embedding_dim
        logger.info(f"Initialized VectorRetriever with embedding dimension {embedding_dim}")
    
    async def retrieve(
        self, 
        query: str,
        query_embedding: List[float],
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        **kwargs
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant documents for a query using vector similarity.
        
        Args:
            query: The query text
            query_embedding: The query embedding
            top_k: Maximum number of results to return
            score_threshold: Minimum similarity score (0-1) for results
            **kwargs: Additional parameters for the retrieval
            
        Returns:
            List of RetrievalResult objects
        """
        # Perform similarity search
        results = await self.vector_store.similarity_search(
            query_embedding=query_embedding,
            k=top_k * 2,  # Get extra results to filter
            filter_func=kwargs.get("filter_func")
        )
        
        # Apply score threshold if provided
        if score_threshold is not None:
            results = [r for r in results if r['score'] >= score_threshold]
        
        # Convert to RetrievalResult objects
        return [
            RetrievalResult(
                content=result['text'],
                metadata=result['metadata'],
                source_type="document",
                score=result['score'],
                chunk_id=result.get('chunk_id')
            )
            for result in results[:top_k]
        ]
    
    def add_documents(
        self,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]]
    ) -> int:
        """
        Add documents to the retriever.
        
        Args:
            chunks: List of document chunks
            embeddings: Corresponding embeddings for the chunks
            
        Returns:
            Number of documents added
        """
        if not chunks or not embeddings:
            return 0
            
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")
        
        self.vector_store.add_documents(chunks, embeddings)
        return len(chunks)
    
    def get_schema(self) -> Dict[str, Any]:
        """Get schema information for the retriever's data source."""
        return {
            "type": "vector_store",
            "embedding_dimension": self.embedding_dim,
            "document_count": self.vector_store.doc_count
        }

class ConnectorRetriever(BaseRetriever):
    """Retriever that uses a connector to fetch data from external sources."""
    
    def __init__(self, connector: BaseConnector):
        """
        Initialize the connector retriever.
        
        Args:
            connector: An instance of a BaseConnector subclass
        """
        self.connector = connector
        self.connected = False
    
    async def connect(self) -> bool:
        """Connect to the data source."""
        if not self.connected:
            self.connected = await self.connector.connect_async()
        return self.connected
    
    async def retrieve(
        self, 
        query: str,
        query_embedding: List[float],
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        **kwargs
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant information using the connector.
        
        Args:
            query: The query text
            query_embedding: The query embedding (not used by all connectors)
            top_k: Maximum number of results to return
            score_threshold: Minimum similarity score (0-1) for results
            **kwargs: Additional parameters for the connector
            
        Returns:
            List of RetrievalResult objects
        """
        if not self.connected:
            await self.connect()
            if not self.connected:
                logger.error("Failed to connect to data source")
                return []

        try:
            # Query the connector
            chunks = await self.connector.search(query=query, **kwargs)

            # Convert to RetrievalResult objects
            results = []
            for chunk in chunks[:top_k]:
                result = RetrievalResult(
                    content=chunk.text,  # Use .text for DocumentChunk
                    metadata=chunk.metadata,
                    source_type=self.connector.__class__.__name__,
                    score=1.0,  # Default score for connector results
                    chunk_id=chunk.chunk_id
                )
                results.append(result)

            return results
            
        except Exception as e:
            logger.error(f"Error in connector retrieval: {e}")
            return []
    
    def get_schema(self) -> Dict[str, Any]:
        """Get schema information from the connector."""
        try:
            return self.connector.get_schema()
        except Exception as e:
            logger.error(f"Error getting schema: {e}")
            return {"error": str(e)}

class HybridRetriever(BaseRetriever):
    """
    A retriever that combines multiple retrievers and merges their results.
    """
    
    def __init__(self, retrievers: List[BaseRetriever]):
        """
        Initialize the hybrid retriever.
        
        Args:
            retrievers: List of retriever instances to use
        """
        self.retrievers = retrievers
    
    async def retrieve(
        self, 
        query: str,
        query_embedding: List[float],
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        **kwargs
    ) -> List[RetrievalResult]:
        """
        Retrieve results from all retrievers and merge them.
        
        Args:
            query: The query text
            query_embedding: The query embedding
            top_k: Maximum number of results to return per retriever
            score_threshold: Minimum similarity score (0-1) for results
            **kwargs: Additional parameters for the retrievers
            
        Returns:
            List of RetrievalResult objects sorted by score
        """
        # Gather results from all retrievers in parallel
        tasks = [
            retriever.retrieve(
                query=query,
                query_embedding=query_embedding,
                top_k=top_k,
                score_threshold=score_threshold,
                **kwargs
            )
            for retriever in self.retrievers
        ]
        
        all_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Flatten and sort results by score
        merged_results = []
        for results in all_results:
            if isinstance(results, Exception):
                logger.error(f"Error in retriever: {results}")
                continue
            merged_results.extend(results)
        
        # Sort by score in descending order
        merged_results.sort(key=lambda x: x.score, reverse=True)
        
        # Apply top_k to the merged results
        return merged_results[:top_k]
    
    def get_schema(self) -> Dict[str, Any]:
        """Get schema information from all retrievers."""
        schemas = {}
        for i, retriever in enumerate(self.retrievers):
            try:
                schema = retriever.get_schema()
                schemas[f"retriever_{i}"] = schema
            except Exception as e:
                logger.error(f"Error getting schema from retriever {i}: {e}")
                schemas[f"retriever_{i}"] = {"error": str(e)}
        
        return {"retrievers": schemas}

def create_retriever(
    retriever_type: str,
    config: Dict[str, Any],
    index_path: Optional[Union[str, Path]] = None,
    embedding_dim: int = 768
) -> BaseRetriever:
    """
    Factory function to create a retriever based on type.
    
    Args:
        retriever_type: Type of retriever to create ("vector", "odoo", etc.)
        config: Configuration dictionary for the retriever
        index_path: Path to the vector index (for vector retrievers)
        embedding_dim: Dimension of the embeddings (for vector retrievers)
        
    Returns:
        An instance of a BaseRetriever subclass
    """
    if retriever_type == "vector":
        return VectorRetriever(
            index_path=index_path,
            embedding_dim=embedding_dim
        )
    
    elif retriever_type == "odoo":
        connector = OdooConnector(config)
        return ConnectorRetriever(connector)
    
    # Add more retriever types as needed
    
    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}")

# For backward compatibility
Retriever = VectorRetriever
