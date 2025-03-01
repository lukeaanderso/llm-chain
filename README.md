# LLM Chain RAG Project

## Overview
This project demonstrates a Retrieval-Augmented Generation (RAG) application using LangChain and LangGraph with Google Vertex AI.

## Prerequisites
- Python 3.9+
- Poetry
- (Optional) Running Chroma vector store server

## What is Chroma?

Chroma is an open-source embedding database designed to make it easy to build AI applications with memory. Key features include:
- Store and retrieve embeddings
- Semantic search capabilities
- Support for various embedding models
- Scalable and performant vector storage

### Starting Chroma Server

#### Using Docker (Recommended)
```bash
# Pull the latest Chroma image
docker pull chromadb/chroma

# Run Chroma server with custom data path
docker run -p 8000:8000 \
    -v ~/data/chroma:/chroma \
    chromadb/chroma
```

#### Using pip
```bash
# Install Chroma
pip install chromadb

# Ensure the data directory exists
mkdir -p ~/data/chroma

# Start Chroma server with custom data path
chroma run \
    --host localhost \
    --port 8000 \
    --path ~/data/chroma/
```

#### Configuring Persistent Storage
When using a custom path, Chroma will:
- Store vector embeddings persistently
- Maintain data between server restarts
- Allow you to specify a dedicated location for your vector database

**Tip:** Always ensure the specified directory exists and you have write permissions.

## Installation
```bash
poetry install
```

## Configuration
### Chroma Vector Store
You can configure the Chroma vector store connection in two ways:

1. Environment Variables:
```bash
export CHROMA_HOST=your-chroma-host
export CHROMA_PORT=your-chroma-port
```

2. Programmatically:
```python
from llm_chain.vectorstore import create_vector_store

vector_store = create_vector_store(
    embeddings, 
    chroma_host='localhost', 
    chroma_port=8000, 
    chroma_ssl=False,
    chroma_collection_name='my_collection'
)
```

### Environment Variables
The following environment variables can be set to configure the application:

```bash
# Chroma server configuration
export CHROMA_HOST=localhost
export CHROMA_PORT=8000

# Authentication for protected websites (e.g., Deephaven docs)
export DEEPHAVEN_USERNAME=your_username
export DEEPHAVEN_PASSWORD=your_password
```

## Project Structure

The project is organized into the following modules:

```
src/llm_chain/
├── crawler.py     # Web crawling functionality
├── llm.py         # LLM and embeddings initialization
├── rag.py         # RAG graph creation and query handling
└── vectorstore.py # Vector store operations
```

### Key Components:

1. **crawler.py**
   - `WebsiteCrawler`: Scrapy-based crawler for advanced web crawling
   - `SimpleCrawler`: Requests/BeautifulSoup-based crawler for simpler needs
   - `load_web_documents`: Utility function to load documents from web URLs

2. **vectorstore.py**
   - `chunk_documents`: Split documents into smaller chunks for embedding
   - `create_vector_store`: Create or connect to a Chroma vector store

3. **llm.py**
   - `get_llm`: Initialize the language model
   - `get_embeddings`: Initialize the embedding model

4. **rag.py**
   - `create_rag_graph`: Create a RAG graph with retrieval and generation steps

### Example Scripts:

- **crawl_deephaven.py**: Crawl Deephaven documentation and store in vector database
- **query_deephaven.py**: Interactive query interface for Deephaven documentation
- **example.py**: Simple example of using the RAG system

## Web Crawling and Authentication

### Crawling Websites
You can now crawl entire websites and extract text for your vector store using the dedicated crawler module:

```python
from llm_chain.crawler import load_web_documents, SimpleCrawler, crawl_website

# Simple document loading
docs = load_web_documents("https://example.com")

# Advanced crawling with Scrapy
docs = crawl_website(
    "https://example.com", 
    allowed_domains=["example.com", "www.example.com"]
)

# Using the SimpleCrawler for more control
crawler = SimpleCrawler(
    base_url="https://example.com",
    max_pages=50,
    delay=1  # 1 second delay between requests
)
docs = crawler.crawl()
```

### Authentication Methods
Support multiple authentication methods:

#### Basic Authentication
```python
# Using load_web_documents
docs = load_web_documents(
    "https://protected-site.com", 
    auth_config={
        'method': 'basic',
        'username': 'your_username',
        'password': 'your_password'
    }
)

# Using SimpleCrawler
crawler = SimpleCrawler(
    base_url="https://protected-site.com",
    username="your_username",
    password="your_password"
)
docs = crawler.crawl()
```

#### Token Authentication
```python
docs = load_web_documents(
    "https://api-site.com", 
    auth_config={
        'method': 'token',
        'token': 'your_access_token'
    }
)
```

#### Session-based Authentication
```python
docs = load_web_documents(
    "https://login-required-site.com", 
    auth_config={
        'method': 'session',
        'login_url': 'https://site.com/login',
        'login_data': {
            'username': 'your_username',
            'password': 'your_password'
        }
    }
)
```

## Usage
```python
from llm_chain.llm import get_llm, get_embeddings
from llm_chain.crawler import load_web_documents
from llm_chain.vectorstore import chunk_documents, create_vector_store
from llm_chain.rag import create_rag_graph

# Initialize components
llm = get_llm()
embeddings = get_embeddings()

# Load and chunk documents
docs = load_web_documents("https://example.com/blog")
chunks = chunk_documents(docs)

# Create vector store (will use Chroma server if configured)
vector_store = create_vector_store(embeddings, chunks)

# Create RAG graph
graph = create_rag_graph(llm, vector_store)

# Query the graph
result = graph.invoke({"question": "Your question here"})
print(result['answer'])
```

## Example Scripts

The project includes several example scripts to demonstrate different functionalities:

### 1. Crawling Documentation

`crawl_deephaven.py` demonstrates how to crawl a documentation website and store the content in a vector database:

```bash
# Set required environment variables
export DEEPHAVEN_USERNAME=your_username
export DEEPHAVEN_PASSWORD=your_password

# Run the crawler
python crawl_deephaven.py
```

### 2. Interactive Query

The project includes an interactive query script that allows you to ask questions about documents stored in your vector database. Here's how to use it:

```python
from llm_chain.llm import get_llm, get_embeddings
from llm_chain.vectorstore import create_vector_store
from llm_chain.rag import create_rag_graph

# Initialize components
llm = get_llm()
embeddings = get_embeddings()

# Connect to existing vector store (without adding new documents)
vector_store = create_vector_store(
    embeddings,
    chroma_host='localhost',
    chroma_port=8000,
    chroma_collection_name='deephaven_docs'
)

# Create RAG graph
graph = create_rag_graph(llm, vector_store)

# Interactive query loop
print("\n===== Documentation Assistant =====")
print("Ask questions about your documents or type 'exit' to quit\n")

while True:
    # Get user question
    question = input("\nYour question: ")
    
    # Exit condition
    if question.lower() in ['exit', 'quit', 'q']:
        print("Goodbye!")
        break
        
    print("\nSearching documentation...")
    
    # Query the graph
    result = graph.invoke({"question": question})
    
    # Display results
    print("\n----- Sources -----")
    for i, doc in enumerate(result["context"], 1):
        print(f"Source {i}: {doc.metadata.get('source', 'Unknown')}")
    
    print("\n----- Answer -----")
    print(result["answer"])
```

### Running the Interactive Query Script

You can run the included query script for Deephaven documentation:

```bash
# Make sure your Chroma server is running
python query_deephaven.py
```

Example interaction:
```
===== Deephaven Documentation Assistant =====
Ask questions about Deephaven or type 'exit' to quit

Your question: What is Deephaven?

Searching documentation...

----- Sources -----
Source 1: https://docs.deephaven.io/latest/Content/index.htm
Source 2: https://docs.deephaven.io/latest/Content/concepts/overview.htm

----- Answer -----
Deephaven is a real-time analytics platform designed to process and analyze streaming and historical data. It allows users to work with both static and dynamic data in a unified environment. Deephaven provides tools for data manipulation, visualization, and analysis, with a focus on high-performance computing for large datasets. It supports various programming interfaces including Python, Java, and a web-based UI, making it accessible for different types of users from data scientists to business analysts.

Your question: When should I use update vs updateView?

Searching documentation...
2025-03-01 17:50:02,396 - INFO - HTTP Request: POST http://localhost:8000/api/v2/tenants/default_tenant/databases/default_database/collections/661ef7d9-0fa9-4ad2-b394-f4efa5b301ce/query "HTTP/1.1 200 OK"

----- Sources -----
Source 1: https://docs.deephaven.io/latest/Content/writeQueries/tableOperations/selection.htm
Source 2: https://docs.deephaven.io/latest/Content/writeQueries/tableOperations/selection.htm
Source 3: https://docs.deephaven.io/latest/Content/writeQueries/tableOperations/selection.htm
Source 4: https://docs.deephaven.io/latest/Content/quickReference/QRG.htm

----- Answer -----
Use `update` when you want to keep all columns from the source table and save the evaluated content into memory, which is recommended for expensive or frequently accessed content. Use `updateView` when you want to keep all columns from the source table but calculate the content on the fly without saving it, which is suitable for fast computations or when accessing a small portion of the data. `updateView` requires less memory because the data is not stored.

--------------------------------------------------

Your question: exit
Goodbye!

WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1740869422.143050   47965 init.cc:232] grpc_wait_for_shutdown_with_timeout() timed out.