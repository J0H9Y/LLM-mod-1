# 🔍 LLM RAG Pipeline with CRM Integration

[![Tests](https://github.com/yourusername/llm-rag-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/llm-rag-pipeline/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/yourusername/llm-rag-pipeline/branch/main/graph/badge.svg)](https://codecov.io/gh/yourusername/llm-rag-pipeline)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready Retrieval-Augmented Generation (RAG) pipeline featuring Gemma 3 4B via Ollama, with seamless CRM/ERP integration for business intelligence.

## 🚀 Key Features

- **Local LLM Integration**: Powered by Gemma 3 4B via Ollama for private, on-device inference
- **CRM/ERP Connectivity**: Native support for HubSpot, with extensible architecture for other platforms
- **Hybrid Retrieval**: Combines vector similarity with structured CRM data for comprehensive responses
- **Modular Design**: Plug-and-play components for documents, databases, and APIs
- **Production-Grade**: Built with testing, monitoring, and scalability in mind

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.9+
- [Ollama](https://ollama.ai/) installed and running
- Gemma 3 4B model: `ollama pull gemma:7b`
- (Optional) HubSpot developer account for CRM integration

### Quick Start

```bash
# 1. Clone the repository
git clone <repository-url>
cd llm-rag-pipeline

# 2. Set up environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env with your configuration
```

### CRM Configuration
For HubSpot integration, add your API credentials to `.env`:
```env
# HubSpot OAuth2 (recommended)
HUBSPOT_ACCESS_TOKEN=your_access_token

# OR API Key (for private apps)
# HUBSPOT_API_KEY=your_api_key
```

## 🏃‍♂️ Quick Start

### Basic Usage

```python
from src.llm import GemmaLLM
from src.rag import QueryRouter, create_retriever
from src.rag.connectors import HubSpotConnector

# Initialize components
llm = GemmaLLM(model="gemma:7b")

# Set up HubSpot connector
hubspot_config = {
    "access_token": "your_token",  # Or use environment variables
    "objects": ["contacts", "companies", "deals"]
}
connector = HubSpotConnector(hubspot_config)

# Create hybrid retriever
retriever = create_retriever("hybrid", connectors=[connector])

# Query the RAG pipeline
query = "Show me top deals closing this month"
results = await retriever.retrieve(query)

# Generate response with LLM
response = await llm.generate(
    f"Answer based on context:\n{results}\n\nQuestion: {query}"
)
print(response)
```

### Using the Demo Script

```bash
# Interactive mode
python scripts/demo_hubspot_rag.py --interactive

# Single query
python scripts/demo_hubspot_rag.py --query "Show me recent high-value deals"
```

## 🏗️ Project Structure

```
llm-rag-pipeline/
├── .github/               # GitHub workflows and templates
├── docs/                  # Documentation (MkDocs)
├── scripts/               # Utility and demo scripts
├── src/                   # Source code
│   ├── llm/              # LLM integration (Gemma via Ollama)
│   ├── rag/              # RAG pipeline components
│   │   ├── connectors/   # CRM/ERP connectors (HubSpot, etc.)
│   │   ├── prompts/      # Prompt templates
│   │   ├── __init__.py   # Core RAG components
│   │   └── ...
│   └── utils/            # Utility functions
├── tests/                # Test suite
│   ├── integration/      # End-to-end tests
│   └── unit/             # Unit tests
├── .env.example          # Example environment variables
├── .github/              # GitHub Actions workflows
├── requirements.txt      # Production dependencies
├── requirements-dev.txt  # Development dependencies
└── README.md            # This file
```

## 🔌 Available Connectors

### HubSpot CRM
- **Features**:
  - OAuth2 and API key authentication
  - Support for Contacts, Companies, Deals
  - Natural language query translation
  - Schema discovery

```python
from src.rag.connectors import HubSpotConnector

config = {
    "access_token": "your_token",
    "objects": ["contacts", "companies", "deals"],
    "properties": {
        "contacts": ["firstname", "lastname", "email"],
        "deals": ["dealname", "amount", "dealstage"]
    }
}
connector = HubSpotConnector(config)
```

## 🧪 Testing

Run the full test suite:

```bash
# Install test dependencies
pip install -r requirements-dev.txt

# Run all tests
pytest

# Run with coverage report
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/connectors/test_hubspot_connector.py -v
```

## 🛠 Development

### Code Style
We use `black` for code formatting and `flake8` for linting:

```bash
# Format code
black .

# Check code style
flake8
```

### Pre-commit Hooks
Set up pre-commit hooks to automatically format and check your code:

```bash
pip install pre-commit
pre-commit install
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

Before submitting a PR, please ensure:
- All tests pass
- Code is properly formatted
- New features include tests
- Documentation is updated

## 📚 Documentation

For detailed documentation, including API reference and usage examples, see our [documentation site](https://your-docs-url.com).

To build documentation locally:

```bash
# Install docs dependencies
pip install mkdocs mkdocs-material

# Serve documentation locally
mkdocs serve
```

## 🚀 Deployment

### Docker

```bash
# Build the image
docker build -t rag-pipeline .

# Run the container
docker run -p 8000:8000 rag-pipeline
```

### Kubernetes
Example deployment files are available in the `deploy/` directory.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ollama](https://ollama.ai/) for making local LLM inference accessible
- [Gemma](https://ai.google.dev/gemma) by Google for the powerful open weights model
- [HubSpot](https://developers.hubspot.com/) for their developer-friendly API
- The open-source community for inspiration and tools
