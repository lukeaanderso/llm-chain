import os
import logging
from llm_chain.llm import get_llm, get_embeddings
from llm_chain.crawler import load_web_documents
from llm_chain.vectorstore import chunk_documents, create_vector_store
from llm_chain.rag import create_rag_graph

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    try:
        # Initialize components
        llm = get_llm()
        embeddings = get_embeddings()

        # Optional: Configure Chroma server (can also use environment variables)
        chroma_host = os.environ.get('CHROMA_HOST', 'localhost')
        chroma_port = os.environ.get('CHROMA_PORT', 8000)

        logger.info(f"Attempting to connect to Chroma server at {chroma_host}:{chroma_port}")

        # Load and chunk documents
        docs = load_web_documents(("https://lilianweng.github.io/posts/2023-06-23-agent/",))
        chunks = chunk_documents(docs)

        # Create vector store with optional server configuration
        vector_store = create_vector_store(
            embeddings, 
            documents=chunks,
            chroma_host=chroma_host,
            chroma_port=int(chroma_port),
            chroma_collection_name='example_collection'
        )

        logger.info("Vector store created successfully")

        # Create RAG graph
        graph = create_rag_graph(llm, vector_store)

        # Query the graph
        result = graph.invoke({"question": "What is Task Decomposition?"})
        
        logger.info("Query processed successfully")
        print(f'Context: {result["context"]}\n\n')
        print(f'Answer: {result["answer"]}')

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    main()
