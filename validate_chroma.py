import chromadb
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def validate_chroma_collection():
    try:
        # Connect to Chroma server
        client = chromadb.HttpClient(host='localhost', port=8000)
        
        # Get the collection we just created
        collection = client.get_collection(name="deephaven_docs")
        
        # Get collection details
        logger.info(f"Collection Name: {collection.name}")
        
        # Count the number of embeddings
        count = collection.count()
        logger.info(f"Total number of documents: {count}")
        
        # Retrieve a few sample documents
        if count > 0:
            # Retrieve the first 5 documents
            results = collection.get(limit=5, include=['documents', 'metadatas'])
            
            logger.info("\nSample Documents:")
            for i, (doc, metadata) in enumerate(zip(results['documents'], results['metadatas']), 1):
                logger.info(f"\nDocument {i}:")
                logger.info(f"Source: {metadata.get('source', 'Unknown')}")
                # Print first 200 characters of the document
                logger.info(f"Preview: {doc[:200]}..." if len(doc) > 200 else f"Preview: {doc}")
        
        return count
    
    except Exception as e:
        logger.error(f"An error occurred while validating Chroma collection: {e}")
        return None

def main():
    validate_chroma_collection()

if __name__ == "__main__":
    main()
