import pytest
from unittest.mock import AsyncMock, MagicMock

from src.rag.retriever_v2 import (
    ConnectorRetriever, HybridRetriever, RetrievalResult, VectorRetriever, create_retriever
)
from src.rag.connectors.base import BaseConnector
from src.rag.ingest import DocumentChunk


@pytest.mark.asyncio
async def test_hybrid_retriever_combines_results():
    """Test that the HybridRetriever correctly combines results from multiple retrievers."""
    # Arrange
    mock_retriever_1 = AsyncMock()
    mock_retriever_1.retrieve.return_value = [
        RetrievalResult(content="doc1", metadata={}, score=0.9, source_type="vector"),
    ]

    mock_retriever_2 = AsyncMock()
    mock_retriever_2.retrieve.return_value = [
        RetrievalResult(content="doc2", metadata={}, score=0.8, source_type="connector"),
    ]

    hybrid_retriever = HybridRetriever(retrievers=[mock_retriever_1, mock_retriever_2])
    query = "test query"

    # Act
    results = await hybrid_retriever.retrieve(query, query_embedding=None)

    # Assert
    assert len(results) == 2
    mock_retriever_1.retrieve.assert_called_once_with(query=query, query_embedding=None, top_k=5, score_threshold=None)
    mock_retriever_2.retrieve.assert_called_once_with(query=query, query_embedding=None, top_k=5, score_threshold=None)
    assert results[0].content == "doc1"
    assert results[1].content == "doc2"


@pytest.mark.asyncio
async def test_hybrid_retriever_handles_empty_results():
    """Test that the HybridRetriever handles cases where one retriever returns no results."""
    # Arrange
    mock_retriever_1 = AsyncMock()
    mock_retriever_1.retrieve.return_value = [
        RetrievalResult(content="doc1", metadata={}, score=0.9, source_type="vector"),
    ]

    mock_retriever_2 = AsyncMock()
    mock_retriever_2.retrieve.return_value = []

    hybrid_retriever = HybridRetriever(retrievers=[mock_retriever_1, mock_retriever_2])
    query = "test query"

    # Act
    results = await hybrid_retriever.retrieve(query, query_embedding=None)

    # Assert
    assert len(results) == 1
    assert results[0].content == "doc1"


@pytest.mark.asyncio
async def test_vector_retriever_retrieves_from_vector_store():
    """Test that VectorRetriever retrieves documents from the vector store."""
    # Arrange
    mock_vector_store = AsyncMock()
    mock_vector_store.similarity_search.return_value = [
        {'text': 'doc1', 'metadata': {}, 'score': 0.9}
    ]

    vector_retriever = VectorRetriever(vector_store=mock_vector_store)
    query = "test query"
    query_embedding = [0.1, 0.2, 0.3]

    # Act
    results = await vector_retriever.retrieve(query, query_embedding=query_embedding)

    # Assert
    mock_vector_store.similarity_search.assert_called_once_with(
        query_embedding=query_embedding, k=10, filter_func=None
    )
    assert len(results) == 1
    assert results[0].content == "doc1"


@pytest.mark.asyncio
async def test_vector_retriever_with_score_threshold():
    """Test that VectorRetriever filters results by score_threshold."""
    # Arrange
    mock_vector_store = AsyncMock()
    mock_vector_store.similarity_search.return_value = [
        {'text': 'doc1', 'metadata': {}, 'score': 0.9},
        {'text': 'doc2', 'metadata': {}, 'score': 0.7}
    ]

    vector_retriever = VectorRetriever(vector_store=mock_vector_store)
    query = "test query"
    query_embedding = [0.1, 0.2, 0.3]

    # Act
    results = await vector_retriever.retrieve(
        query, query_embedding=query_embedding, score_threshold=0.8
    )

    # Assert
    assert len(results) == 1
    assert results[0].content == "doc1"


def test_vector_retriever_add_documents():
    """Test adding documents to the VectorRetriever."""
    # Arrange
    mock_vector_store = MagicMock()
    vector_retriever = VectorRetriever(vector_store=mock_vector_store)
    chunks = [MagicMock(), MagicMock()]
    embeddings = [[0.1], [0.2]]

    # Act
    count = vector_retriever.add_documents(chunks, embeddings)

    # Assert
    assert count == 2
    mock_vector_store.add_documents.assert_called_once_with(chunks, embeddings)


def test_vector_retriever_add_documents_mismatch_error():
    """Test that add_documents raises ValueError for mismatched lengths."""
    # Arrange
    vector_retriever = VectorRetriever(vector_store=MagicMock())
    chunks = [MagicMock()]
    embeddings = [[0.1], [0.2]]

    # Act & Assert
    with pytest.raises(ValueError):
        vector_retriever.add_documents(chunks, embeddings)


@pytest.mark.asyncio
async def test_hybrid_retriever_handles_retriever_exception():
    """Test that HybridRetriever handles exceptions from a sub-retriever."""
    mock_retriever_1 = AsyncMock()
    mock_retriever_1.retrieve.return_value = [
        RetrievalResult(content="doc1", metadata={}, score=0.9, source_type="vector"),
    ]
    mock_retriever_2 = AsyncMock()
    mock_retriever_2.retrieve.side_effect = Exception("Retriever failed")

    hybrid_retriever = HybridRetriever(retrievers=[mock_retriever_1, mock_retriever_2])
    results = await hybrid_retriever.retrieve("test", query_embedding=None)

    assert len(results) == 1
    assert results[0].content == "doc1"


def test_hybrid_retriever_get_schema_handles_exception():
    """Test that HybridRetriever handles exceptions when getting a sub-schema."""
    mock_retriever_1 = MagicMock()
    mock_retriever_1.get_schema.return_value = {"type": "vector"}
    mock_retriever_2 = MagicMock()
    mock_retriever_2.get_schema.side_effect = Exception("Schema failed")

    hybrid_retriever = HybridRetriever(retrievers=[mock_retriever_1, mock_retriever_2])
    schema = hybrid_retriever.get_schema()

    assert "retriever_0" in schema["retrievers"]
    assert "retriever_1" in schema["retrievers"]
    assert schema["retrievers"]["retriever_0"] == {"type": "vector"}
    assert "error" in schema["retrievers"]["retriever_1"]


def test_vector_retriever_add_documents_empty():
    """Test that add_documents handles empty lists gracefully."""
    # Arrange
    mock_vector_store = MagicMock()
    vector_retriever = VectorRetriever(vector_store=mock_vector_store)

    # Act
    count = vector_retriever.add_documents([], [])

    # Assert
    assert count == 0
    mock_vector_store.add_documents.assert_not_called()


def test_vector_retriever_get_schema():
    """Test getting the schema from the VectorRetriever."""
    # Arrange
    mock_vector_store = MagicMock()
    mock_vector_store.doc_count = 100
    vector_retriever = VectorRetriever(vector_store=mock_vector_store, embedding_dim=128)

    # Act
    schema = vector_retriever.get_schema()

    # Assert
    assert schema == {
        "type": "vector_store",
        "embedding_dimension": 128,
        "document_count": 100
    }


@pytest.mark.asyncio
async def test_connector_retriever_retrieves_from_connector():
    """Test that ConnectorRetriever retrieves documents from the connector."""
    # Arrange
    mock_connector = AsyncMock()
    mock_chunk = MagicMock(spec=DocumentChunk)
    mock_chunk.text = 'doc1'
    mock_chunk.metadata = {'source': 'api'}
    mock_chunk.chunk_id = 'chunk1'
    mock_connector.search.return_value = [mock_chunk]

    connector_retriever = ConnectorRetriever(connector=mock_connector)
    query = "test query"

    # Act
    results = await connector_retriever.retrieve(query, query_embedding=None)

    # Assert
    mock_connector.search.assert_called_once_with(query=query)
    assert len(results) == 1
    assert results[0].content == "doc1"
    assert results[0].source_type == "AsyncMock"


@pytest.mark.asyncio
async def test_connector_retriever_retrieve_connection_fails():
    """Test that ConnectorRetriever handles connection failure."""
    mock_connector = AsyncMock()
    mock_connector.connect.return_value = False
    connector_retriever = ConnectorRetriever(connector=mock_connector)

    results = await connector_retriever.retrieve("test", query_embedding=None)

    assert results == []
    mock_connector.connect.assert_called_once()
    mock_connector.search.assert_not_called()


@pytest.mark.asyncio
async def test_connector_retriever_retrieve_search_fails():
    """Test that ConnectorRetriever handles a search exception."""
    mock_connector = AsyncMock()
    mock_connector.connect.return_value = True
    mock_connector.search.side_effect = Exception("Search failed")
    connector_retriever = ConnectorRetriever(connector=mock_connector)

    results = await connector_retriever.retrieve("test", query_embedding=None)

    assert results == []
    mock_connector.search.assert_called_once_with(query="test")


def test_connector_retriever_get_schema_fails():
    """Test that ConnectorRetriever handles a get_schema exception."""
    mock_connector = MagicMock()
    mock_connector.get_schema.side_effect = Exception("Schema failed")
    connector_retriever = ConnectorRetriever(connector=mock_connector)

    schema = connector_retriever.get_schema()

    assert "error" in schema
    assert schema["error"] == "Schema failed"


def test_connector_retriever_get_schema():
    """Test getting the schema from the ConnectorRetriever."""
    # Arrange
    mock_connector = MagicMock()
    mock_connector.get_schema.return_value = {"type": "connector", "name": "TestConnector"}
    connector_retriever = ConnectorRetriever(connector=mock_connector)

    # Act
    schema = connector_retriever.get_schema()

    # Assert
    assert schema == {"type": "connector", "name": "TestConnector"}
    mock_connector.get_schema.assert_called_once()


def test_create_retriever_vector_store(mocker):
    """Test creating a VectorRetriever using the factory."""
    mock_vr = mocker.patch('src.rag.retriever_v2.VectorRetriever')
    config = {}
    _ = create_retriever("vector", config=config, index_path="/tmp/test", embedding_dim=128)
    mock_vr.assert_called_once_with(index_path="/tmp/test", embedding_dim=128)


def test_create_retriever_odoo(mocker):
    """Test creating a ConnectorRetriever for Odoo using the factory."""
    mock_cr = mocker.patch('src.rag.retriever_v2.ConnectorRetriever')
    mock_oc = mocker.patch('src.rag.retriever_v2.OdooConnector')
    config = {"url": "localhost", "db": "test", "username": "admin", "password": "password"}
    _ = create_retriever("odoo", config=config)
    mock_oc.assert_called_once_with(config)
    mock_cr.assert_called_once_with(mock_oc.return_value)


def test_create_retriever_unknown_type():
    """Test that creating a retriever with an unknown type raises a ValueError."""
    with pytest.raises(ValueError, match="Unknown retriever type: unknown_type"):
        create_retriever("unknown_type", config={})
