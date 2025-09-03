"""
Vector store implementation using FAISS for efficient similarity search.
"""
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import faiss
import numpy as np

from src.rag.ingest import DocumentChunk
from src.utils.logging import log


class VectorStore:
    """
    A vector store that uses FAISS for efficient similarity search.
    """

    def __init__(
        self, index_path: Optional[Union[str, Path]] = None, dimension: int = 768
    ):
        """
        Initialize the vector store.

        Args:
            index_path: Path to save/load the FAISS index and metadata.
            dimension: Dimension of the embeddings.
        """
        self.dimension = dimension
        self.index_path = Path(index_path) if index_path else None
        self.metadata_path = (
            Path(f"{index_path}.meta") if index_path else None
        )

        # Initialize empty index and metadata
        self.index = faiss.IndexIDMap(faiss.IndexFlatL2(dimension))
        self.metadata: Dict[str, Dict[str, Any]] = {}
        self.doc_count = 0

        # Initialize metrics
        self.reset_metrics()

        # Load existing index if path is provided and exists
        if self.index_path and self.index_path.exists():
            self._load_from_disk()

    def reset_metrics(self) -> None:
        """Resets the metrics counters."""
        self.metrics = {
            "add_documents": {"count": 0, "errors": 0},
            "delete_document": {"count": 0, "errors": 0},
            "similarity_search": {"count": 0, "errors": 0, "latencies": []},
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Returns the collected metrics, including average latency."""
        search_metrics = self.metrics["similarity_search"]
        if search_metrics["latencies"]:
            avg_latency = sum(search_metrics["latencies"]) / len(
                search_metrics["latencies"]
            )
            self.metrics["similarity_search"]["average_latency_ms"] = round(
                avg_latency, 2
            )
        return self.metrics

    def add_documents(
        self, chunks: List[DocumentChunk], embeddings: List[List[float]]
    ) -> None:
        """
        Add document chunks with their embeddings to the vector store.

        Args:
            chunks: List of document chunks.
            embeddings: List of corresponding embeddings (list of floats).
        """
        if not chunks or not embeddings:
            return

        try:
            if len(chunks) != len(embeddings):
                raise ValueError(
                    "Number of chunks must match number of embeddings"
                )

            embeddings_array = np.array(embeddings, dtype="float32")
            start_idx = self.doc_count
            ids_to_add = np.arange(
                start_idx, start_idx + len(chunks), dtype="int64"
            )

            self.index.add_with_ids(embeddings_array, ids_to_add)

            for i, chunk in enumerate(chunks):
                doc_id = str(ids_to_add[i])
                self.metadata[doc_id] = {
                    "text": chunk.text,
                    "chunk_id": chunk.chunk_id,
                    "metadata": chunk.metadata,
                }

            self.doc_count += len(chunks)
            log.info(
                f"Added {len(chunks)} documents to vector store. "
                f"Total: {self.doc_count}"
            )
            self.metrics["add_documents"]["count"] += len(chunks)

        except Exception as e:
            log.error(f"Error adding documents: {e}", exc_info=True)
            self.metrics["add_documents"]["errors"] += 1
            raise

    def similarity_search(
        self, query_embedding: List[float], k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find the k most similar documents to the query embedding.

        Args:
            query_embedding: The query embedding.
            k: Number of results to return.

        Returns:
            List of dictionaries containing document text, metadata, and score.
        """
        start_time = time.monotonic()
        self.metrics["similarity_search"]["count"] += 1
        try:
            if self.doc_count == 0:
                return []

            query = np.array([query_embedding], dtype="float32")
            distances, indices = self.index.search(
                query, min(k * 2, self.doc_count)
            )

            results = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx < 0:
                    continue

                doc_id = str(idx)
                if doc_id not in self.metadata:
                    log.warning(f"Document ID {doc_id} not found in metadata")
                    continue

                metadata = self.metadata[doc_id]
                results.append(
                    {
                        "text": metadata["text"],
                        "metadata": metadata["metadata"],
                        "chunk_id": metadata["chunk_id"],
                        "distance": float(distance),
                        "score": 1.0 / (1.0 + float(distance)),
                    }
                )

                if len(results) >= k:
                    break

            results.sort(key=lambda x: x["score"], reverse=True)
            return results

        except Exception as e:
            log.error(f"Error during similarity search: {e}", exc_info=True)
            self.metrics["similarity_search"]["errors"] += 1
            return []
        finally:
            latency = (time.monotonic() - start_time) * 1000  # in ms
            self.metrics["similarity_search"]["latencies"].append(latency)

    def save(self, path: Optional[Union[str, Path]] = None) -> None:
        """
        Save the vector store to disk.

        Args:
            path: Optional path to save the index. Uses self.index_path if None.
        """
        if path is None and self.index_path is None:
            raise ValueError("No path provided for saving the vector store")

        save_path = Path(path) if path else self.index_path
        meta_path = Path(f"{save_path}.meta")

        faiss.write_index(self.index, str(save_path))

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "dimension": self.dimension,
                    "doc_count": self.doc_count,
                    "metadata": self.metadata,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        log.info(
            f"Saved vector store to {save_path} with {self.doc_count} documents"
        )

    def _load_from_disk(self) -> None:
        """Load the vector store from disk."""
        if not self.index_path or not self.metadata_path:
            return

        if not self.index_path.exists() or not self.metadata_path.exists():
            log.warning("Vector store files not found, starting with empty store")
            return

        self.index = faiss.read_index(str(self.index_path))

        with open(self.metadata_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.dimension = data["dimension"]
            self.doc_count = data["doc_count"]
            self.metadata = data["metadata"]

        log.info(
            f"Loaded vector store with {self.doc_count} documents from "
            f"{self.index_path}"
        )

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a document by its ID.

        Args:
            doc_id: Document ID

        Returns:
            Document metadata or None if not found
        """
        return self.metadata.get(str(doc_id))

    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document from the vector store.

        Args:
            doc_id: Document ID to delete

        Returns:
            True if document was deleted, False if not found
        """
        doc_id_str = str(doc_id)
        try:
            if doc_id_str not in self.metadata:
                log.warning(
                    f"Attempted to delete non-existent document ID: {doc_id_str}"
                )
                return False

            id_to_remove = np.array([int(doc_id_str)], dtype=np.int64)
            remove_count = self.index.remove_ids(id_to_remove)

            if remove_count == 0:
                log.warning(
                    f"Doc ID {doc_id_str} in metadata but not in FAISS index."
                )

            del self.metadata[doc_id_str]
            self.doc_count = self.index.ntotal

            log.info(
                f"Deleted document {doc_id_str}. New count: {self.doc_count}"
            )
            self.metrics["delete_document"]["count"] += 1
            return True

        except Exception as e:
            log.error(
                f"Error deleting document {doc_id_str}: {e}", exc_info=True
            )
            self.metrics["delete_document"]["errors"] += 1
            return False
