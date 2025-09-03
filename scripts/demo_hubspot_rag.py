#!/usr/bin/env python3
"""
Demo script showing RAG pipeline with HubSpot integration.

Example usage:
    python scripts/demo_hubspot_rag.py --query "Show me top deals closing this month"
"""
import os
import sys
import json
import logging
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path
import argparse
from dotenv import load_dotenv

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.llm.gemma_wrapper import GemmaLLM
from src.rag.connectors.hubspot_connector import HubSpotConnector
from src.rag.retriever_v2 import (ConnectorRetriever, HybridRetriever, RetrievalResult)
from src.rag.prompts import prompt_renderer
from src.utils.logging import setup_logging

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)

class RAGPipeline:
    """RAG Pipeline with HubSpot integration."""
    
    def __init__(self):
        """Initialize the RAG pipeline components."""
        self.llm = GemmaLLM()
        self.connector = None
        self.retriever = None
    
    async def initialize(self):
        """Initialize all components."""
        logger.info("Initializing RAG pipeline...")
        
        # Initialize HubSpot connector
        hubspot_config = {
            "access_token": os.getenv("HUBSPOT_ACCESS_TOKEN"),
            "api_key": os.getenv("HUBSPOT_API_KEY"),
            "batch_size": 5,
            "objects": ["deals", "contacts", "companies"],
            "properties": {
                "deals": ["dealname", "amount", "dealstage", "closedate", "pipeline"],
                "contacts": ["firstname", "lastname", "email", "phone"],
                "companies": ["name", "domain", "industry"]
            }
        }
        
        self.connector = HubSpotConnector(hubspot_config)
        # The ConnectorRetriever will handle the connection

        # Initialize retriever with a connector
        connector_retriever = ConnectorRetriever(connector=self.connector)
        self.retriever = HybridRetriever(retrievers=[connector_retriever])
        
        logger.info("RAG pipeline initialized successfully")
    
    async def query(self, query: str) -> str:
        """
        Process a query through the RAG pipeline.
        
        Args:
            query: Natural language query
            
        Returns:
            Generated response
        """
        if not self.retriever:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")
        
        logger.info(f"Processing query: {query}")
        
        # Step 1: Retrieve relevant context
        # The query_embedding is not needed for the ConnectorRetriever
        retrieval_results = await self.retriever.retrieve(query)
        
        # Step 2: Format context for the prompt
        context = self._format_retrieval_results(retrieval_results)
        
        # Step 3: Generate response using the LLM
        response = await self._generate_response(query, context)
        
        return response
    
    def _format_retrieval_results(self, results: List[RetrievalResult]) -> str:
        """Format retrieval results into a string for the prompt."""
        if not results:
            return "No relevant information found."
            
        formatted = ["## Retrieved Information\n"]
        
        for i, result in enumerate(results, 1):
            source_type = result.metadata.get('source', 'unknown')
            object_type = result.metadata.get('object_type', 'document')
            
            formatted.append(f"### Source {i}: {source_type.upper()} ({object_type})")
            formatted.append(f"**Relevance Score:** {result.score:.3f}")
            
            # Add source-specific metadata
            if 'url' in result.metadata:
                formatted.append(f"**URL:** {result.metadata['url']}")
                
            # Add the content
            formatted.append("\n" + result.content + "\n")
        
        return "\n".join(formatted)
    
    async def _generate_response(self, query: str, context: str) -> str:
        """Generate a response using the LLM."""
        # Determine which template to use based on query intent
        template_name = self._determine_template(query)
        
        # Render the prompt with the context and query
        prompt = prompt_renderer.render(
            f"{template_name}.jinja2",
            context={"query": query, "context": context}
        )
        
        logger.debug(f"Generated prompt:\n{prompt}")
        
        # Generate response
        response = self.llm.generate(prompt)
        
        # Format the final response
        return self._format_response(response, context)
    
    def _determine_template(self, query: str) -> str:
        """Determine which prompt template to use based on the query."""
        query_lower = query.lower()
        
        if any(term in query_lower for term in ["deal", "sale", "pipeline", "opportunity"]):
            return "sales_analysis"
        elif any(term in query_lower for term in ["contact", "person", "email", "phone"]):
            return "customer_analysis"
        elif any(term in query_lower for term in ["company", "business", "organization"]):
            return "company_analysis"
        else:
            return "generic_query"
    
    def _format_response(self, response: str, context: str) -> str:
        """Format the final response with context and sources."""
        # Simple formatting - in a real app, you might want to make this more sophisticated
        return f"""## Response
{response}

## Sources Used
{context}"""

async def main():
    """Main entry point."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="RAG Pipeline with HubSpot Integration")
    parser.add_argument("--query", type=str, help="Query to process")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Initialize pipeline
    pipeline = RAGPipeline()
    await pipeline.initialize()
    
    if args.query:
        # Process a single query
        response = await pipeline.query(args.query)
        print("\n" + "="*80)
        print(response)
        print("="*80 + "\n")
    elif args.interactive:
        # Interactive mode
        print("RAG Pipeline with HubSpot Integration")
        print("Type 'exit' or 'quit' to exit\n")
        
        while True:
            try:
                query = input("\nEnter your query: ").strip()
                if query.lower() in ('exit', 'quit'):
                    break
                    
                if not query:
                    continue
                    
                print("\nProcessing...\n")
                response = await pipeline.query(query)
                print("\n" + "="*80)
                print(response)
                print("="*80 + "\n")
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                logger.error(f"Error: {e}")
                print(f"\nError: {e}\n")
    else:
        # Default demo queries
        demo_queries = [
            "Show me top deals closing this month",
            "Find contacts from companies in the technology industry",
            "Summarize our sales pipeline"
        ]
        
        print("RAG Pipeline with HubSpot Integration - Demo Mode\n")
        print("Example queries:")
        for i, query in enumerate(demo_queries, 1):
            print(f"{i}. {query}")
        
        while True:
            try:
                choice = input("\nSelect a query (1-3) or type your own: ").strip()
                
                if choice.lower() in ('exit', 'quit'):
                    break
                    
                if not choice:
                    continue
                    
                # Check if it's a number
                if choice.isdigit() and 1 <= int(choice) <= len(demo_queries):
                    query = demo_queries[int(choice) - 1]
                else:
                    query = choice
                
                print(f"\nRunning query: {query}\n")
                response = await pipeline.query(query)
                print("\n" + "="*80)
                print(response)
                print("="*80 + "\n")
                
                # Show the demo menu again
                print("\nExample queries:")
                for i, q in enumerate(demo_queries, 1):
                    print(f"{i}. {q}")
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                logger.error(f"Error: {e}")
                print(f"\nError: {e}\n")

if __name__ == "__main__":
    asyncio.run(main())
