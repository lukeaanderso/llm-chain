import os
import logging
from llm_chain.llm import get_llm, get_embeddings
from llm_chain.vectorstore import create_vector_store
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
        logger.info("Initializing LLM and embeddings...")
        llm = get_llm()
        embeddings = get_embeddings()

        # Configure Chroma server
        chroma_host = os.environ.get('CHROMA_HOST', 'localhost')
        chroma_port = os.environ.get('CHROMA_PORT', 8000)
        collection_name = 'deephaven_docs'

        logger.info(f"Connecting to Chroma server at {chroma_host}:{chroma_port}, collection: {collection_name}")

        # Create vector store connection (without documents - we're using existing collection)
        vector_store = create_vector_store(
            embeddings,
            chroma_host=chroma_host,
            chroma_port=int(chroma_port),
            chroma_collection_name=collection_name
        )

        logger.info("Connected to vector store successfully")

        # Create RAG graph
        logger.info("Creating RAG graph...")
        graph = create_rag_graph(llm, vector_store)

        # Interactive query loop
        print("\n===== Deephaven Documentation Assistant =====")
        print("Ask questions about Deephaven or type 'exit' to quit\n")

        while True:
            # Get user question
            question = input("\nYour question: ")
            
            # Exit condition
            if question.lower() in ['exit', 'quit', 'q']:
                print("Goodbye!")
                break
                
            if not question.strip():
                continue
                
            print("\nSearching documentation...")
            
            # Query the graph
            result = graph.invoke({"question": question})
            
            # Display results
            print("\n----- Sources -----")
            for i, doc in enumerate(result["context"], 1):
                print(f"Source {i}: {doc.metadata.get('source', 'Unknown')}")
            
            print("\n----- Answer -----")
            print(result["answer"])
            print("\n" + "-" * 50)

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    main()
