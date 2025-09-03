"""Configuration and fixtures for tests."""
import os
import pytest
from unittest.mock import patch, MagicMock

# Set environment variables for testing
os.environ["LOG_LEVEL"] = "DEBUG"
os.environ["LOG_FILE"] = "test.log"

# Test configuration
TEST_CONFIG = {
    "hubspot": {
        "access_token": "test_token",
        "batch_size": 5,
        "objects": ["contacts", "companies", "deals"],
        "properties": {
            "contacts": ["firstname", "lastname", "email", "phone"],
            "companies": ["name", "domain", "industry"],
            "deals": ["dealname", "amount", "dealstage", "closedate"]
        }
    }
}

@pytest.fixture(autouse=True)
def mock_environment_vars():
    """Mock environment variables for testing."""
    with patch.dict(os.environ, {
        "HUBSPOT_ACCESS_TOKEN": "test_token",
        "LOG_LEVEL": "DEBUG"
    }):
        yield

@pytest.fixture
def mock_llm():
    """Mock the LLM for testing."""
    with patch('src.llm.gemma_wrapper.GemmaLLM') as mock_llm:
        mock_instance = mock_llm.return_value
        mock_instance.generate.return_value = "Test response"
        yield mock_instance

@pytest.fixture
def mock_requests():
    """Mock requests for API calls."""
    with patch('requests.Session') as mock_session:
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_response.status_code = 200
        mock_session.return_value.get.return_value = mock_response
        mock_session.return_value.post.return_value = mock_response
        yield mock_session
