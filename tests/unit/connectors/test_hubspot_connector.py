"""Unit tests for HubSpot connector."""
import pytest
from unittest.mock import patch, MagicMock

from src.rag.connectors.hubspot_connector import HubSpotConnector
from src.rag.ingest import DocumentChunk

# Test data
MOCK_CONTACT = {
    "id": "123",
    "properties": {
        "firstname": "John",
        "lastname": "Doe",
        "email": "john.doe@example.com",
    },
    "created_at": "2023-01-01T00:00:00Z",
    "updated_at": "2023-01-15T00:00:00Z",
}

MOCK_COMPANY = {
    "id": "456",
    "properties": {
        "name": "Acme Inc.",
        "domain": "acme.com",
        "industry": "Technology",
    },
    "created_at": "2022-01-01T00:00:00Z",
    "updated_at": "2023-01-15T00:00:00Z",
}

MOCK_DEAL = {
    "id": "789",
    "properties": {
        "dealname": "Enterprise Plan",
        "amount": "10000",
        "dealstage": "closedwon",
    },
    "created_at": "2023-01-01T00:00:00Z",
    "updated_at": "2023-01-15T00:00:00Z",
}


@pytest.fixture
def hubspot_connector():
    """Fixture for HubSpotConnector."""
    config = {"access_token": "fake_token", "portal_id": "12345"}
    return HubSpotConnector(config)


class TestHubSpotConnector:
    """Test cases for HubSpotConnector."""

    @patch("src.rag.connectors.hubspot_connector.HubSpot")
    def test_connect_success(self, mock_hubspot, hubspot_connector):
        """Test successful connection to HubSpot."""
        # Arrange
        mock_client = mock_hubspot.return_value
        mock_client.crm.owners.get_by_id.return_value = MagicMock()

        # Act
        result = hubspot_connector.connect()

        # Assert
        assert result is True
        assert hubspot_connector.connected is True
        mock_hubspot.assert_called_once_with(access_token="fake_token")
        mock_client.crm.owners.get_by_id.assert_called_once_with("me")

    @patch("src.rag.connectors.hubspot_connector.HubSpot")
    def test_connect_failure(self, mock_hubspot, hubspot_connector):
        """Test failed connection to HubSpot."""
        # Arrange
        mock_client = mock_hubspot.return_value
        mock_client.crm.owners.get_by_id.side_effect = Exception("API Error")

        # Act
        result = hubspot_connector.connect()

        # Assert
        assert result is False
        assert hubspot_connector.connected is False

    @patch("src.rag.connectors.hubspot_connector.HubSpot")
    def test_get_chunks(self, mock_hubspot, hubspot_connector):
        """Test getting all chunks from specified objects."""
        # Arrange
        hubspot_connector.connect()  # Establishes the mocked client
        mock_client = hubspot_connector.client
        mock_client.crm.contacts.get_all.return_value = [MagicMock(to_dict=lambda: MOCK_CONTACT)]
        mock_client.crm.companies.get_all.return_value = [MagicMock(to_dict=lambda: MOCK_COMPANY)]
        mock_client.crm.deals.get_all.return_value = [MagicMock(to_dict=lambda: MOCK_DEAL)]

        # Act
        chunks = hubspot_connector.get_chunks()

        # Assert
        assert len(chunks) == 3
        assert chunks[0].metadata["object_type"] == "contact"
        assert chunks[1].metadata["object_type"] == "company"
        assert chunks[2].metadata["object_type"] == "deal"
        mock_client.crm.contacts.get_all.assert_called_once()
        mock_client.crm.companies.get_all.assert_called_once()
        mock_client.crm.deals.get_all.assert_called_once()

    @patch("src.rag.connectors.hubspot_connector.HubSpot")
    def test_get_chunks_specific_object(self, mock_hubspot, hubspot_connector):
        """Test getting chunks for only one object type."""
        # Arrange
        hubspot_connector.connect()
        mock_client = hubspot_connector.client
        mock_client.crm.contacts.get_all.return_value = [MagicMock(to_dict=lambda: MOCK_CONTACT)]

        # Act
        chunks = hubspot_connector.get_chunks(objects=["contacts"])

        # Assert
        assert len(chunks) == 1
        assert chunks[0].metadata["object_type"] == "contact"
        mock_client.crm.contacts.get_all.assert_called_once()
        mock_client.crm.companies.get_all.assert_not_called()
        mock_client.crm.deals.get_all.assert_not_called()

    def test_document_chunk_conversion(self, hubspot_connector):
        """Test conversion of HubSpot objects to DocumentChunks."""
        # Test contact
        contact_chunk = hubspot_connector._contact_to_chunk(MOCK_CONTACT, ["firstname", "lastname"])
        assert isinstance(contact_chunk, DocumentChunk)
        assert "John" in contact_chunk.text
        assert contact_chunk.metadata["object_type"] == "contact"

        # Test company
        company_chunk = hubspot_connector._company_to_chunk(MOCK_COMPANY, ["name", "industry"])
        assert "Acme Inc." in company_chunk.text
        assert company_chunk.metadata["object_type"] == "company"

        # Test deal
        deal_chunk = hubspot_connector._deal_to_chunk(MOCK_DEAL, ["dealname", "amount"])
        assert "Enterprise Plan" in deal_chunk.text
        assert deal_chunk.metadata["object_type"] == "deal"
