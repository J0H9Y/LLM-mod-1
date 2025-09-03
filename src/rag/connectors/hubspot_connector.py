"""
HubSpot connector for the RAG pipeline.
"""
from typing import Any, Dict, List

from hubspot import HubSpot

from src.rag.connectors.base import APIConnector
from src.rag.ingest import DocumentChunk
from src.utils.logging import log


class HubSpotConnector(APIConnector):
    """
    Connector for HubSpot CRM.

    Example config:
    {
        "access_token": "your_access_token",
        "objects": ["contacts", "companies", "deals"],  # Objects to query
        "batch_size": 100,  # Number of records to fetch per batch
        "properties": {  # Properties to fetch for each object type
            "contacts": ["firstname", "lastname", "email", "phone", "company"],
            "companies": ["name", "domain", "industry", "annualrevenue"],
            "deals": ["dealname", "amount", "dealstage", "closedate"]
        }
    }
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the HubSpot connector."""
        super().__init__(config)
        self.client = None
        self.batch_size = config.get("batch_size", 100)
        self.properties = config.get(
            "properties",
            {
                "contacts": ["firstname", "lastname", "email"],
                "companies": ["name", "domain", "industry"],
                "deals": ["dealname", "amount", "dealstage"],
            },
        )
        self.objects = config.get("objects", ["contacts", "companies", "deals"])
        self.access_token = config.get("access_token")

        if not self.access_token:
            raise ValueError("access_token must be provided in config")

    def connect(self) -> bool:
        """Connect to HubSpot API."""
        try:
            self.client = HubSpot(access_token=self.access_token)
            # Test the connection by making a simple API call
            # Get account info to verify the token is valid
            account_info = self.client.crm.contacts.basic_api.get_page(limit=1)
            self.connected = True
            log.info("Successfully connected to HubSpot")
            return True
        except Exception as e:
            log.error(f"Failed to connect to HubSpot: {e}", exc_info=True)
            self.connected = False
            return False

    def disconnect(self) -> None:
        """Disconnect from HubSpot."""
        self.client = None
        self.connected = False
        log.info("Disconnected from HubSpot")

    def get_chunks(self, **kwargs) -> List[DocumentChunk]:
        """
        Load data from HubSpot and return a list of document chunks.

        Args:
            **kwargs: Can specify 'objects' to override config.

        Returns:
            List of DocumentChunk objects.
        """
        if not self.connected:
            if not self.connect():
                return []

        objects_to_fetch = kwargs.get("objects", self.objects)
        chunks = []

        for obj_type in objects_to_fetch:
            try:
                log.info(f"Fetching chunks for object type: {obj_type}")
                if obj_type == "contacts":
                    obj_chunks = self._get_all_contacts(**kwargs)
                elif obj_type == "companies":
                    obj_chunks = self._get_all_companies(**kwargs)
                elif obj_type == "deals":
                    obj_chunks = self._get_all_deals(**kwargs)
                else:
                    log.warning(f"Unsupported object type: {obj_type}")
                    continue
                chunks.extend(obj_chunks)
                log.info(f"Fetched {len(obj_chunks)} chunks for {obj_type}")
            except Exception as e:
                log.error(f"Error fetching {obj_type}: {e}", exc_info=True)
                continue

        return chunks

    async def search(self, query: str, **kwargs) -> List[DocumentChunk]:
        """
        Search for relevant data based on a query.
        
        Args:
            query: The search query
            **kwargs: Additional parameters
            
        Returns:
            List of DocumentChunk objects
        """
        # For now, we'll return all chunks and let the retriever filter them
        # In a more sophisticated implementation, you could use the query to filter
        # or prioritize certain objects/types
        return self.get_chunks(**kwargs)

    def _get_all_contacts(self, **kwargs) -> List[DocumentChunk]:
        """Retrieve all contacts with specified properties."""
        properties = self.properties.get("contacts", [])
        limit = kwargs.get("limit", 10)  # Default limit for testing
        results = self.client.crm.contacts.basic_api.get_page(limit=limit, properties=properties)
        return [
            self._contact_to_chunk(contact.to_dict(), properties)
            for contact in results.results
        ]

    def _get_all_companies(self, **kwargs) -> List[DocumentChunk]:
        """Retrieve all companies with specified properties."""
        properties = self.properties.get("companies", [])
        limit = kwargs.get("limit", 10)  # Default limit for testing
        results = self.client.crm.companies.basic_api.get_page(limit=limit, properties=properties)
        return [
            self._company_to_chunk(company.to_dict(), properties)
            for company in results.results
        ]

    def _get_all_deals(self, **kwargs) -> List[DocumentChunk]:
        """Retrieve all deals with specified properties."""
        properties = self.properties.get("deals", [])
        limit = kwargs.get("limit", 10)  # Default limit for testing
        results = self.client.crm.deals.basic_api.get_page(limit=limit, properties=properties)
        return [
            self._deal_to_chunk(deal.to_dict(), properties) for deal in results.results
        ]

    def _contact_to_chunk(
        self, contact: Dict[str, Any], properties: List[str]
    ) -> DocumentChunk:
        """Convert a HubSpot contact to a DocumentChunk."""
        props = contact.get("properties", {})
        content_lines = ["# Contact Information"]
        for prop in properties:
            if prop in props and props[prop] is not None:
                content_lines.append(f"- {prop}: {props[prop]}")

        metadata = {
            "source": "hubspot",
            "object_type": "contact",
            "id": contact.get("id"),
            "created_at": contact.get("created_at"),
            "updated_at": contact.get("updated_at"),
            "url": f'https://app.hubspot.com/contacts/{self.config.get("portal_id", "")}/contact/{contact.get("id")}',
        }

        return DocumentChunk(
            text="\n".join(content_lines),
            chunk_id=f'hubspot-contact-{contact.get("id")}',
            metadata=metadata,
        )

    def _company_to_chunk(
        self, company: Dict[str, Any], properties: List[str]
    ) -> DocumentChunk:
        """Convert a HubSpot company to a DocumentChunk."""
        props = company.get("properties", {})
        content_lines = ["# Company Information"]
        for prop in properties:
            if prop in props and props[prop] is not None:
                content_lines.append(f"- {prop}: {props[prop]}")

        metadata = {
            "source": "hubspot",
            "object_type": "company",
            "id": company.get("id"),
            "created_at": company.get("created_at"),
            "updated_at": company.get("updated_at"),
            "url": f'https://app.hubspot.com/contacts/{self.config.get("portal_id", "")}/company/{company.get("id")}',
        }

        return DocumentChunk(
            text="\n".join(content_lines),
            chunk_id=f'hubspot-company-{company.get("id")}',
            metadata=metadata,
        )

    def _deal_to_chunk(
        self, deal: Dict[str, Any], properties: List[str]
    ) -> DocumentChunk:
        """Convert a HubSpot deal to a DocumentChunk."""
        props = deal.get("properties", {})
        content_lines = ["# Deal Information"]
        for prop in properties:
            if prop in props and props[prop] is not None:
                content_lines.append(f"- {prop}: {props[prop]}")

        metadata = {
            "source": "hubspot",
            "object_type": "deal",
            "id": deal.get("id"),
            "created_at": deal.get("created_at"),
            "updated_at": deal.get("updated_at"),
            "url": f'https://app.hubspot.com/contacts/{self.config.get("portal_id", "")}/deal/{deal.get("id")}',
        }

        return DocumentChunk(
            text="\n".join(content_lines),
            chunk_id=f'hubspot-deal-{deal.get("id")}',
            metadata=metadata,
        )
