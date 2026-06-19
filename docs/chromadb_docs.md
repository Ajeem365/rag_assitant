# ChromaDB — Vector Database Documentation

## What is ChromaDB?

ChromaDB is an open-source embedding database (vector store) designed for AI applications. It stores embeddings and their associated metadata, enabling fast similarity search for RAG systems, semantic search, and recommendation engines.

---

## Installation

```bash
pip install chromadb
```

---

## Basic Usage

```python
import chromadb

# In-memory client (no persistence)
client = chromadb.Client()

# Persistent client (saves to disk)
client = chromadb.PersistentClient(path="./chroma_db")
```

---

## Collections

A collection is analogous to a table in a relational database:

```python
# Create or get a collection
collection = client.get_or_create_collection(
    name="my_documents",
    metadata={"hnsw:space": "cosine"}  # Use cosine similarity
)
```

---

## Adding Documents

```python
collection.add(
    documents=["This is doc 1", "This is doc 2"],
    metadatas=[{"source": "file1.md"}, {"source": "file2.md"}],
    ids=["id1", "id2"]  # Must be unique
)
```

---

## Querying

```python
results = collection.query(
    query_texts=["What is RAG?"],
    n_results=3,
    include=["documents", "metadatas", "distances"]
)
```

---

## LangChain Integration

```python
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create from documents
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",
    collection_name="rag_docs"
)

# Load existing
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
    collection_name="rag_docs"
)

# Search
docs = vectorstore.similarity_search("query", k=5)
docs_with_scores = vectorstore.similarity_search_with_score("query", k=5)

# MMR search
docs = vectorstore.max_marginal_relevance_search("query", k=5, fetch_k=20)
```

---

## Deduplication

To avoid re-indexing the same content, assign deterministic IDs:

```python
import hashlib

def get_chunk_id(content: str, source: str) -> str:
    return hashlib.md5(f"{source}::{content}".encode()).hexdigest()

# ChromaDB will skip documents with duplicate IDs
vectorstore.add_documents(chunks, ids=[get_chunk_id(c.page_content, c.metadata["source"]) for c in chunks])
```

---

## Inspecting the Collection

```python
# Count documents
count = collection.count()

# Get all documents
result = collection.get(include=["metadatas", "documents"])

# Delete a collection
client.delete_collection("my_documents")
```

---

## Similarity Metrics

ChromaDB supports three distance metrics:
- `cosine` (default for text): measures angle between vectors
- `l2`: Euclidean distance
- `ip`: Inner product

For text embeddings, cosine similarity is almost always the right choice.
