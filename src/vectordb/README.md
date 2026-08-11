# Vector Database Package

This package handles all vector store operations for the hospital chatbot's RAG system.

## Structure

```
src/vectordb/
├── __init__.py       # Package exports
└── manager.py        # VectorStoreManager class
```

## Components

### VectorStoreManager (`manager.py`)

Manages the ChromaDB vector store for knowledge base retrieval.

**Key Features:**
- Header-based markdown chunking strategy
- Automatic metadata enrichment (department, section, chunk_id)
- Persistent storage in `.chroma_db/`
- Semantic similarity search

**Usage:**

```python
from src.vectordb import VectorStoreManager

# Initialize
vector_store = VectorStoreManager()
vector_store.initialize()  # Loads existing or creates new

# Search
results = vector_store.similarity_search("cardiology hours", k=3)

# Search with scores
results_with_scores = vector_store.similarity_search_with_score("cardiology hours", k=3)

# Force rebuild
vector_store.initialize(force_reload=True)
```

## Chunking Strategy

**Primary Split: Markdown Headers**
- `#` → Document title
- `##` → Department/Section (Cardiology, Appointments, etc.)
- `###` → Subsections (General Hours, Appointment Contact, etc.)

**Secondary Split: Size-Based**
- Chunks larger than 1000 chars are split further
- 100 char overlap for context continuity

**Metadata Enrichment:**
- `section`: Header hierarchy
- `department`: Detected department name (if applicable)
- `chunk_id`: Sequential chunk identifier

## Dependencies

- `langchain` - Text splitting utilities
- `langchain-openai` - OpenAI embeddings
- `langchain-community` - ChromaDB integration
- `chromadb` - Vector database

## Configuration

Configured via `src/config.py`:
- `kb_dir`: Knowledge base directory path
- `base_dir`: Base directory for `.chroma_db/`
- OpenAI API key for embeddings (from environment)

