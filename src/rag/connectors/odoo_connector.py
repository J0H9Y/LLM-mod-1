"""
Odoo connector for the RAG pipeline.
"""
import xmlrpc.client
from typing import Dict, List, Any, Optional
import logging
import json
from datetime import datetime
from ..ingest import DocumentChunk
from .base import APIConnector
from ...utils.logging import log as logger

class OdooConnector(APIConnector):
    """
    Connector for Odoo ERP system.
    
    Example config:
    {
        "url": "https://your-odoo-instance.com",
        "db": "your_database",
        "username": "your_username",
        "password": "your_password",
        "models": ["sale.order", "account.invoice"],  # Models to query
        "chunk_size": 10  # Number of records per chunk
    }
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.models = config.get("models", [])
        self.chunk_size = config.get("chunk_size", 10)
        self.common = None
        self.models_proxy = None
        self.uid = None

    def connect(self) -> bool:
        """Connect to Odoo instance."""
        try:
            # Initialize connections
            self.common = xmlrpc.client.ServerProxy(f"{self.base_url}/xmlrpc/2/common")
            self.uid = self.common.authenticate(
                self.config["db"], 
                self.config["username"], 
                self.config["password"], 
                {}
            )
            
            self.models_proxy = xmlrpc.client.ServerProxy(f"{self.base_url}/xmlrpc/2/object")
            self.connected = True
            logger.info("Successfully connected to Odoo")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Odoo: {e}")
            self.connected = False
            return False

    def disconnect(self) -> None:
        """Disconnect from Odoo."""
        self.common = None
        self.models_proxy = None
        self.uid = None
        self.connected = False
        logger.info("Disconnected from Odoo")

    def get_chunks(self, query: str, **kwargs) -> List[DocumentChunk]:
        """
        Query Odoo for data matching the query.
        
        Args:
            query: Natural language query
            **kwargs: Additional parameters like model names, filters, etc.
            
        Returns:
            List of DataChunk objects
        """
        if not self.connected:
            self.connect()
            if not self.connected:
                return []
        
        # Determine which models to query
        models_to_query = kwargs.get("models") or self.models
        if not models_to_query:
            models_to_query = self._detect_relevant_models(query)
        
        chunks = []
        
        # Query each model
        for model_name in models_to_query:
            try:
                # Get model fields for better context
                fields = self._get_model_fields(model_name)
                
                # Build search domain based on query
                domain = self._build_search_domain(query, fields)
                
                # Search for records
                record_ids = self.models_proxy.execute_kw(
                    self.config["db"], 
                    self.uid, 
                    self.config["password"],
                    model_name, 
                    'search', 
                    [domain],
                    {'limit': self.chunk_size}
                )
                
                if not record_ids:
                    continue
                
                # Get record details
                fields_to_fetch = self._get_relevant_fields(model_name, query, fields)
                records = self.models_proxy.execute_kw(
                    self.config["db"], 
                    self.uid, 
                    self.config["password"],
                    model_name, 
                    'read', 
                    [record_ids],
                    {'fields': fields_to_fetch}
                )
                
                # Convert records to chunks
                for record in records:
                    chunk = self._record_to_chunk(model_name, record, fields)
                    chunks.append(chunk)
                
            except Exception as e:
                logger.error(f"Error querying Odoo model {model_name}: {e}")
                continue
        
        return chunks
    
    def _detect_relevant_models(self, query: str) -> List[str]:
        """Determine which models are relevant to the query."""
        # Simple keyword-based model detection
        query_lower = query.lower()
        relevant_models = []
        
        model_keywords = {
            "sale.order": ["sale", "order", "quote", "quotation"],
            "account.invoice": ["invoice", "bill", "payment"],
            "crm.lead": ["lead", "opportunity", "prospect"],
            "project.task": ["task", "project", "milestone"],
            "hr.employee": ["employee", "staff", "team member"]
        }
        
        for model, keywords in model_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                relevant_models.append(model)
        
        return relevant_models or list(model_keywords.keys())[:2]  # Default to first 2 models if none match
    
    def _get_model_fields(self, model_name: str) -> Dict[str, Any]:
        """Get fields information for a model."""
        try:
            return self.models_proxy.execute_kw(
                self.config["db"], 
                self.uid, 
                self.config["password"],
                model_name, 
                'fields_get', 
                [],
                {'attributes': ['string', 'type', 'relation']}
            )
        except Exception as e:
            logger.warning(f"Could not get fields for model {model_name}: {e}")
            return {}
    
    def _build_search_domain(self, query: str, fields: Dict[str, Any]) -> List[Any]:
        """Build a search domain based on the query and available fields."""
        # This is a simplified version - in production, you'd want to use NLP
        # to better understand the query and build more precise domains
        
        domain = []
        query_terms = query.lower().split()
        
        # Look for specific field references in the query
        for field_name, field_info in fields.items():
            field_label = field_info.get('string', '').lower()
            
            # Check if any query term matches the field label
            for term in query_terms:
                if term in field_label and len(term) > 2:  # Only consider terms longer than 2 chars
                    domain.append((field_name, 'ilike', term))
        
        # If no field-specific terms found, do a general search
        if not domain:
            domain = [('name', 'ilike', query)]
        
        return domain
    
    def _get_relevant_fields(self, model_name: str, query: str, fields: Dict[str, Any]) -> List[str]:
        """Determine which fields to fetch based on the query."""
        # Always include these fields
        default_fields = ['id', 'name', 'create_date', 'write_date']
        
        # Add fields that match query terms
        query_terms = query.lower().split()
        relevant_fields = set(default_fields)
        
        for field_name, field_info in fields.items():
            field_label = field_info.get('string', '').lower()
            
            # Include fields that match query terms
            if any(term in field_label for term in query_terms if len(term) > 2):
                relevant_fields.add(field_name)
            
            # Include fields that are likely to be useful
            field_type = field_info.get('type')
            if field_type in ['char', 'text', 'html', 'selection']:
                relevant_fields.add(field_name)
        
        return list(relevant_fields)
    
    def _record_to_chunk(self, model_name: str, record: Dict[str, Any], fields: Dict[str, Any]) -> DocumentChunk:
        """Convert an Odoo record to a DocumentChunk."""
        
        # Format the record as a readable string
        lines = [f"# {model_name.upper()}: {record.get('name', 'Unnamed')}", ""]
        
        # Add fields in a readable format
        for field_name, value in record.items():
            if field_name == 'id' or not value:
                continue
                
            field_info = fields.get(field_name, {})
            field_label = field_info.get('string', field_name.replace('_', ' ').title())
            
            # Format the value based on its type
            if isinstance(value, (list, tuple)):
                if value and isinstance(value[0], (int, float)) and len(value) > 1:
                    # Handle many2many and one2many fields
                    lines.append(f"- {field_label}: {len(value)} items")
                elif value and isinstance(value[0], (int, float)):
                    # Handle many2one fields
                    lines.append(f"- {field_label}: ID {value[0]}")
                else:
                    lines.append(f"- {field_label}: {', '.join(str(v) for v in value)}")
            elif isinstance(value, dict):
                lines.append(f"- {field_label}:")
                for k, v in value.items():
                    lines.append(f"  - {k}: {v}")
            else:
                # Format dates and datetimes
                if field_info.get('type') == 'datetime' and value:
                    try:
                        dt = datetime.strptime(value, '%Y-%m-%d %H:%M:%S')
                        value = dt.strftime('%Y-%m-%d %H:%M')
                    except (ValueError, TypeError):
                        pass
                elif field_info.get('type') == 'date' and value:
                    try:
                        dt = datetime.strptime(value, '%Y-%m-%d')
                        value = dt.strftime('%Y-%m-%d')
                    except (ValueError, TypeError):
                        pass
                
                lines.append(f"- {field_label}: {value}")
        
        # Create metadata
        metadata = {
            "source": "odoo",
            "model": model_name,
            "record_id": record.get('id'),
            "create_date": record.get('create_date'),
            "update_date": record.get('write_date')
        }
        
        return DocumentChunk(
            text="\n".join(lines),
            metadata=metadata,
            chunk_id=f"odoo_{model_name}_{record.get('id')}"
        )
    
    def get_schema(self) -> Dict[str, Any]:
        """Get schema information for all configured models."""
        if not self.connected:
            return {}
        
        schema = {}
        
        for model_name in (self.models or []):
            try:
                fields = self._get_model_fields(model_name)
                schema[model_name] = {
                    "fields": [
                        {"name": name, "type": info.get('type', 'unknown'), 
                         "label": info.get('string', name)}
                        for name, info in fields.items()
                    ]
                }
            except Exception as e:
                logger.error(f"Error getting schema for model {model_name}: {e}")
                continue
        
        return schema
