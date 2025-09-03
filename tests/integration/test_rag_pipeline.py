"""Integration tests for the full RAG pipeline with HubSpot."""
import os
import pytest
import asyncio
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

# Test data
MOCK_RETRIEVAL_RESULTS = [
    {
        "content": "# Contact Information\n- firstname: John\n- lastname: Doe\n- email: john.doe@example.com",
        "metadata": {
            "source": "hubspot",
            "object_type": "contact",
            "id": "123",
            "url": "https://app.hubspot.com/contacts/123/contact/123"
        },
        "score": 0.95
    },
    {
        "content": "# Deal Information\n- dealname: Enterprise Plan\n- amount: 10000\n- dealstage: closedwon",
        "metadata": {
            "source": "hubspot",
            "object_type": "deal",
            "id": "789",
            "url": "https://app.hubspot.com/contacts/123/deal/789"
        },
        "score": 0.85
    }
]

# Fixtures

@pytest.fixture
def mock_retriever():
    class MockRetriever:
        async def retrieve(self, query: str, query_embedding: list = None, **kwargs):
            from src.rag.retriever_v2 import RetrievalResult
            return [
                RetrievalResult(
                    content=result["content"],
                    metadata=result["metadata"],
                    score=result["score"],
                    source_type=result["metadata"]["source"]
                )
                for result in MOCK_RETRIEVAL_RESULTS
            ]
    
    return MockRetriever()

@pytest.fixture
def mock_llm_response():
    return "Here's the information you requested about John Doe and the Enterprise Plan deal."

@pytest.fixture
def mock_gemma_llm(mock_llm_response):
    with patch('src.llm.gemma_wrapper.GemmaLLM') as mock_llm:
        mock_instance = mock_llm.return_value
        mock_instance.generate.return_value = mock_llm_response
        yield mock_instance

@pytest.fixture
def mock_hubspot_connector():
    with patch('src.rag.connectors.hubspot_connector.HubSpotConnector') as mock_connector:
        mock_instance = mock_connector.return_value
        mock_instance.connected = True
        mock_instance.get_schema.return_value = {"hubspot": {"contacts": {}, "deals": {}}}
        
        # Mock the query method
        async def mock_query(query, **kwargs):
            from src.rag.connectors.base import DataChunk
            return [
                DataChunk(
                    content=result["content"],
                    metadata=result["metadata"],
                    source_type="hubspot",
                    chunk_id=f"hubspot_{result['metadata']['object_type']}_{result['metadata']['id']}"
                )
                for result in MOCK_RETRIEVAL_RESULTS
            ]
        
        mock_instance.query = mock_query
        yield mock_instance

@pytest.fixture
def rag_pipeline(mock_gemma_llm, mock_retriever, mock_hubspot_connector):
    from scripts.demo_hubspot_rag import RAGPipeline
    
    class TestRAGPipeline(RAGPipeline):
        def __init__(self):
            self.llm = mock_gemma_llm
            self.retriever = mock_retriever
            self.connector = mock_hubspot_connector
    
    return TestRAGPipeline()

# Tests
class TestRAGPipelineIntegration:
    """Integration tests for the RAG pipeline with HubSpot."""
    
    @pytest.mark.asyncio
    async def test_full_rag_flow(self, rag_pipeline, mock_llm_response):
        """Test the full RAG pipeline flow."""
        # Test query
        query = "Show me information about John Doe and his deals"
        
        # Process the query
        response = await rag_pipeline.query(query)
        
        # Verify the response
        assert response is not None
        assert isinstance(response, str)
        
        # Check if the LLM was called with the right prompt
        rag_pipeline.llm.generate.assert_called_once()
        
        # The response should include the mock LLM response
        assert mock_llm_response in response
        
        # The response should include the retrieved context
        for result in MOCK_RETRIEVAL_RESULTS:
            assert result["content"] in response
    
    @pytest.mark.asyncio
    async def test_format_retrieval_results(self, rag_pipeline):
        """Test formatting of retrieval results."""
        from src.rag.retriever_v2 import RetrievalResult
        
        # Create test retrieval results
        results = [
            RetrievalResult(
                content=result["content"],
                metadata=result["metadata"],
                score=result["score"],
                source_type=result["metadata"]["source"]
            )
            for result in MOCK_RETRIEVAL_RESULTS
        ]
        
        # Format the results
        formatted = rag_pipeline._format_retrieval_results(results)
        
        # Check the formatted output
        assert "## Retrieved Information" in formatted
        
        # Check if all results are included
        for result in MOCK_RETRIEVAL_RESULTS:
            assert result["metadata"]["object_type"].capitalize() in formatted
            assert str(result["score"])[:4] in formatted  # Check score is included
            assert result["content"] in formatted
    
    @pytest.mark.asyncio
    async def test_determine_template(self, rag_pipeline):
        """Test template determination based on query intent."""
        # Test sales-related queries
        assert rag_pipeline._determine_template("show me deals") == "sales_analysis"
        assert rag_pipeline._determine_template("sales pipeline") == "sales_analysis"
        
        # Test contact-related queries
        assert rag_pipeline._determine_template("find contacts") == "customer_analysis"
        assert rag_pipeline._determine_template("person info") == "customer_analysis"
        
        # Test company-related queries
        assert rag_pipeline._determine_template("company details") == "company_analysis"
        
        # Test fallback to generic query
        assert rag_pipeline._determine_template("random query") == "generic_query"
    
    @pytest.mark.asyncio
    async def test_generate_response(self, rag_pipeline, mock_llm_response):
        """Test response generation with mock LLM."""
        query = "Test query"
        context = "Test context"
        
        # Generate response
        response = await rag_pipeline._generate_response(query, context)
        
        # Check if the LLM was called with the right arguments
        rag_pipeline.llm.generate.assert_called_once()
        
        # The response should include the mock LLM response
        assert mock_llm_response in response
    
    @pytest.mark.asyncio
    async def test_format_response(self, rag_pipeline):
        """Test formatting of the final response."""
        llm_response = "Test response"
        context = "Test context"
        
        # Format the response
        response = rag_pipeline._format_response(llm_response, context)
        
        # Check the formatted output
        assert "## Response" in response
        assert "## Sources Used" in response
        assert llm_response in response
        assert context in response
