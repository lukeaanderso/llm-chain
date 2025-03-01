import os
import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llm_chain.llm import get_llm, get_embeddings
from llm_chain.vectorstore import chunk_documents, create_vector_store
from llm_chain.crawler import SimpleCrawler

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

        # Chroma server configuration
        chroma_host = os.environ.get('CHROMA_HOST', 'localhost')
        chroma_port = os.environ.get('CHROMA_PORT', 8000)

        # Authentication details from environment variables
        base_url = "https://docs.deephaven.io/latest/Content/index.htm"
        username = os.environ.get('DEEPHAVEN_USERNAME', '')
        password = os.environ.get('DEEPHAVEN_PASSWORD', '')

        logger.info(f"Starting crawler at: {base_url}")
        
        # Initialize and run the crawler
        crawler = SimpleCrawler(
            base_url=base_url,
            username=username if username else None,
            password=password if password else None,
            max_pages=50,  # Limit to 50 pages
            delay=1  # 1 second delay between requests
        )
        
        # Crawl the website
        docs = crawler.crawl()
        
        if not docs:
            logger.error("No documents were extracted during crawling")
            return

        # Chunk the documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200
        )
        chunks = chunk_documents(docs)

        logger.info(f"Extracted {len(chunks)} document chunks from {len(docs)} pages")

        # Create vector store with Chroma server
        vector_store = create_vector_store(
            embeddings, 
            documents=chunks,
            chroma_host=chroma_host,
            chroma_port=int(chroma_port),
            chroma_collection_name='deephaven_docs'
        )

        logger.info("Vector store created successfully")

        # Print first chunk details
        if chunks:
            print("Sample document chunk:")
            print(chunks[0].page_content[:500])  # Print first 500 chars of first chunk

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    main()
