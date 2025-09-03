"""
Base classes for data connectors.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional

from src.utils.logging import log
from src.rag.ingest import DocumentChunk


class BaseConnector(ABC):
    """Base class for all data connectors."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the connector with configuration.

        Args:
            config: Configuration dictionary specific to the connector
        """
        self.config = config
        self.connected = False
        log.info(f"Initialized {self.__class__.__name__}")

    @abstractmethod
    def connect(self) -> bool:
        """
        Establish connection to the data source.

        Returns:
            bool: True if connection was successful, False otherwise
        """
        pass

    async def connect_async(self) -> bool:
        """
        Async version of connect method.
        
        Returns:
            bool: True if connection was successful, False otherwise
        """
        return self.connect()

    @abstractmethod
    def disconnect(self) -> None:
        """Close the connection to the data source."""
        pass

    @abstractmethod
    def get_chunks(self, **kwargs) -> List[DocumentChunk]:
        """
        Load data from the source and return a list of document chunks.

        Args:
            **kwargs: Additional parameters for loading data

        Returns:
            List of DocumentChunk objects
        """
        pass

    def get_schema(self) -> Optional[Dict[str, Any]]:
        """
        Get the schema of the data source, if applicable.

        Returns:
            Dictionary describing the schema or None
        """
        log.warning(f"get_schema() not implemented for {self.__class__.__name__}")
        return None

    def __enter__(self):
        """Support for context manager protocol."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Support for context manager protocol."""
        self.disconnect()


class DocumentConnector(BaseConnector):
    """Base class for document-based connectors (e.g., file system)."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.source_type = "document"


class APIConnector(BaseConnector):
    """Base class for API-based connectors."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.source_type = "api"
        # For example, using the 'requests' library
        self.session = None
        self.base_url = config.get("base_url", "")
        self.headers = config.get("headers", {})

    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """
        Make an HTTP request.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            **kwargs: Additional arguments for the request

        Returns:
            Response data as a dictionary
        """
        raise NotImplementedError("Subclasses must implement _make_request")
