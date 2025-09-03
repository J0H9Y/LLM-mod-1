"""Retriever module for the RAG pipeline.

Handles document retrieval from a VectorStore using vector similarity search.
"""
from typing import List, Dict, Any, Optional, Union, Callable
from pathlib import Path

from src.utils.logging import log
from .ingest import DocumentChunk, load_document, load_directory
from .vector_store import VectorStore


class Retriever:
    """Handles document retrieval using vector similarity search."""

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        index_path: Optional[Union[str, Path]] = None,
        embedding_dim: int = 768,
    ):
        """Initialize the retriever.

        Args:
            vector_store: Pre-initialized vector store. If None, a new one is created.
            index_path: Path to load/save the vector store index.
            embedding_dim: Dimension of the embeddings.
        """
        if vector_store:
            self.vector_store = vector_store
        else:
            self.vector_store = VectorStore(index_path=index_path, dimension=embedding_dim)

        self.embedding_dim = self.vector_store.dimension
        log.info(
            "Initialized retriever",
            embedding_dim=self.embedding_dim,
            index_path=str(self.vector_store.index_path),
        )

    def add_documents(
        self,
        source: Union[str, Path, List[DocumentChunk]],
        embeddings: Optional[List[List[float]]] = None,
        **kwargs,
    ) -> int:
        """Add documents to the retriever's vector store.

        Args:
            source: A file path, directory path, or list of DocumentChunk objects.
            embeddings: Pre-computed embeddings for the documents.
            **kwargs: Additional arguments for the document loader.

        Returns:
            The number of documents added.

        Raises:
            ValueError: If the source is invalid or if embeddings are not provided.
        """
        if isinstance(source, (str, Path)):
            source_path = Path(source)
            if source_path.is_file():
                chunks = load_document(source_path, **kwargs)
            elif source_path.is_dir():
                chunks = load_directory(source_path, **kwargs)
            else:
                raise ValueError(f"Source path does not exist: {source_path}")
        elif isinstance(source, list) and all(isinstance(x, DocumentChunk) for x in source):
            chunks = source
        else:
            raise TypeError(
                "Source must be a file/directory path or a list of DocumentChunk objects."
            )

        if not chunks:
            log.warning("No documents found or loaded from source", source=str(source))
            return 0

        if embeddings is None:
            raise ValueError("Embeddings must be provided to add documents.")

        if len(chunks) != len(embeddings):
            raise ValueError("The number of document chunks must match the number of embeddings.")

        self.vector_store.add_documents(chunks, embeddings)
        log.info(f"Added {len(chunks)} documents to the vector store.")
        return len(chunks)

    def retrieve(
        self,
        query: str,
        query_embedding: List[float],
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        filter_func: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for a given query.

        Args:
            query: The query text (for logging purposes).
            query_embedding: The embedding vector for the query.
            top_k: The maximum number of results to return.
            score_threshold: The minimum similarity score (0-1) for results.
            filter_func: An optional function to filter results based on metadata.

        Returns:
            A list of dictionaries, each containing document text, metadata, and score.
        """
        results = self.vector_store.similarity_search(
            query_embedding=query_embedding,
            k=top_k * 2,  # Retrieve more to allow for filtering
            filter_func=filter_func,
        )

        if score_threshold is not None:
            results = [r for r in results if r["score"] >= score_threshold]

        final_results = results[:top_k]

        log.info(
            f"Retrieved {len(final_results)} documents for query",
            query=f"{query[:80]}...",
            top_k=top_k,
        )
        return final_results

    def save_index(self, path: Optional[Union[str, Path]] = None) -> None:
        """Save the vector store index to disk.

        Args:
            path: Path to save the index. If None, uses the path from initialization.
        """
        self.vector_store.save(path)

    def get_document(self, doc_id: str) -> Optional[DocumentChunk]:
        """Retrieve a document by its ID.

        Args:
            doc_id: The ID of the document to retrieve.

        Returns:
            The document chunk or None if not found.
        """
        return self.vector_store.get_document(doc_id)

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document from the vector store.

        Args:
            doc_id: The ID of the document to delete.

        Returns:
            True if the document was deleted, False otherwise.
        """
        return self.vector_store.delete_document(doc_id)
