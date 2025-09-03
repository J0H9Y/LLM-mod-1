import streamlit as st
import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv

from src.llm.gemma_wrapper import GemmaLLM
from src.rag.retriever_v2 import VectorRetriever, ConnectorRetriever, HybridRetriever
from src.rag.ingest import load_directory
from src.rag.connectors.hubspot_connector import HubSpotConnector
from src.rag.connectors.odoo_connector import OdooConnector
from src.utils.feedback import log_feedback

# Load environment variables
load_dotenv()

st.title("RAG Pipeline Interface")

@st.cache_resource
def initialize_rag_pipeline():
    llm = GemmaLLM()
    
    # Initialize different retrievers based on data source
    retrievers = {}
    
    # 1. Document retriever (for local docs)
    doc_retriever = VectorRetriever(embedding_dim=3584)
    docs_path = Path(__file__).parent / "docs"
    if docs_path.exists():
        chunks = load_directory(docs_path)
        embeddings = [llm.generate_embeddings(chunk.text) for chunk in chunks]
        
        # Filter out chunks where embedding generation failed
        valid_chunks = [chunk for i, chunk in enumerate(chunks) if embeddings[i] is not None]
        valid_embeddings = [emb for emb in embeddings if emb is not None]
        
        added = doc_retriever.add_documents(valid_chunks, valid_embeddings)
        if added > 0:
            retrievers["Documents"] = doc_retriever
    
    # 2. HubSpot CRM connector
    hubspot_token = os.getenv("HUBSPOT_ACCESS_TOKEN")
    if hubspot_token:
        try:
            hubspot_config = {
                "access_token": hubspot_token,
                "batch_size": 5,
                "objects": ["deals", "contacts", "companies"],
                "properties": {
                    "deals": ["dealname", "amount", "dealstage", "closedate", "pipeline"],
                    "contacts": ["firstname", "lastname", "email", "phone"],
                    "companies": ["name", "domain", "industry"]
                }
            }
            hubspot_connector = HubSpotConnector(hubspot_config)
            if hubspot_connector.connect():
                retrievers["CRM"] = ConnectorRetriever(connector=hubspot_connector)
                st.success("✅ Connected to HubSpot CRM")
            else:
                st.warning("⚠️ Failed to connect to HubSpot CRM")
        except Exception as e:
            st.error(f"❌ Error initializing HubSpot: {e}")
    
    # 3. Odoo ERP connector
    odoo_url = os.getenv("ODOO_URL")
    odoo_db = os.getenv("ODOO_DB")
    odoo_username = os.getenv("ODOO_USERNAME")
    odoo_password = os.getenv("ODOO_PASSWORD")
    
    if all([odoo_url, odoo_db, odoo_username, odoo_password]):
        try:
            odoo_config = {
                "url": odoo_url,
                "db": odoo_db,
                "username": odoo_username,
                "password": odoo_password,
                "models": ["sale.order", "account.invoice", "res.partner"],
                "chunk_size": 10
            }
            odoo_connector = OdooConnector(odoo_config)
            if odoo_connector.connect():
                retrievers["ERP"] = ConnectorRetriever(connector=odoo_connector)
                st.success("✅ Connected to Odoo ERP")
            else:
                st.warning("⚠️ Failed to connect to Odoo ERP")
        except Exception as e:
            st.error(f"❌ Error initializing Odoo: {e}")
    
    return llm, retrievers

# Define feedback log file
FEEDBACK_LOG_FILE = Path(__file__).parent / "feedback_log.jsonl"

llm, retrievers = initialize_rag_pipeline()


# Input field for the user's question
question = st.text_input("Ask a business question:")

# Dropdown to select data source
available_sources = list(retrievers.keys())
if not available_sources:
    st.error("❌ No data sources available. Please check your configuration.")
    st.stop()

source = st.selectbox("Select data source:", available_sources)

# Initialize session state for feedback
if 'last_query' not in st.session_state:
    st.session_state.last_query = None
if 'last_context' not in st.session_state:
    st.session_state.last_context = None
if 'last_response' not in st.session_state:
    st.session_state.last_response = None

if st.button("Get Answer"):
    if question:
        st.write(f"Querying {source} with: '{question}'")
        # Run RAG pipeline
        with st.spinner('Thinking...'):
            # Get the appropriate retriever for the selected source
            retriever = retrievers[source]
            
            try:
                # 2. Retrieve context (always pass query_embedding; some retrievers ignore it)
                query_embedding = llm.generate_embeddings(question) or []
                retrieval_results = asyncio.run(
                    retriever.retrieve(question, query_embedding, top_k=3)
                )
                
                if retrieval_results:
                    retrieved_context = "\n\n---\n\n".join(
                        [res.content for res in retrieval_results]
                    )
                else:
                    retrieved_context = "No relevant information found in the selected data source."

                # 3. Generate response
                prompt = f"""Use the following context to answer the question.

Context:
{retrieved_context}

Question: {question}

Answer:"""
                response_data = llm.query(prompt)
                llm_response = response_data.get("response", "No response from LLM.")

                st.subheader("Retrieved Context")
                st.text_area("", retrieved_context, height=150)
                
            except Exception as e:
                st.error(f"Error during retrieval: {e}")
                llm_response = f"Error: {str(e)}"
                retrieved_context = "Error occurred during data retrieval."

            st.subheader("LLM Response")
            st.write(llm_response)

            # Store results for feedback
            st.session_state.last_query = question
            st.session_state.last_context = retrieved_context
            st.session_state.last_response = llm_response

            # Feedback section
            st.subheader("Feedback")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("👍 Thumbs Up"):
                    if st.session_state.last_query:
                        log_feedback(
                            FEEDBACK_LOG_FILE,
                            st.session_state.last_query,
                            st.session_state.last_context,
                            st.session_state.last_response,
                            'thumbs_up'
                        )
                        st.success("Feedback received!")
                    else:
                        st.warning("Please perform a query first.")
            with col2:
                if st.button("👎 Thumbs Down"):
                    if st.session_state.last_query:
                        log_feedback(
                            FEEDBACK_LOG_FILE,
                            st.session_state.last_query,
                            st.session_state.last_context,
                            st.session_state.last_response,
                            'thumbs_down'
                        )
                        st.error("Feedback received!")
                    else:
                        st.warning("Please perform a query first.")
            
            feedback_text = st.text_area("Additional comments:")
            if st.button("Submit Feedback"):
                if st.session_state.last_query:
                    log_feedback(
                        FEEDBACK_LOG_FILE,
                        st.session_state.last_query,
                        st.session_state.last_context,
                        st.session_state.last_response,
                        'comment',
                        comment=feedback_text
                    )
                    st.success("Thank you for your feedback!")
                else:
                    st.warning("Please perform a query first.")
    else:
        st.warning("Please enter a question.")
