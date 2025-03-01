import os
import bs4
import logging
from typing import List, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import chromadb
from langchain_core.documents import Document
from src.llm_chain.crawler import load_web_documents as crawler_load_documents

def chunk_documents(documents, chunk_size=1000, chunk_overlap=200):
    """
    Split documents into chunks.
    
    Args:
        documents (list): List of documents to chunk
        chunk_size (int): Size of each document chunk
        chunk_overlap (int): Number of characters to overlap between chunks
    
    Returns:
        List of document chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(documents)

def create_vector_store(embeddings, documents=None, 
                        chroma_host=None, 
                        chroma_port=None, 
                        chroma_ssl=False, 
                        chroma_collection_name="default_collection"):
    """
    Create a vector store, optionally connecting to a Chroma server.
    
    Args:
        embeddings: Embedding function
        documents (list, optional): Documents to add to vector store
        chroma_host (str, optional): Hostname of Chroma server
        chroma_port (int, optional): Port of Chroma server
        chroma_ssl (bool, optional): Whether to use SSL for Chroma connection
        chroma_collection_name (str, optional): Name of the Chroma collection
    
    Returns:
        Chroma vector store
    """
    # Check for Chroma server configuration via environment variables
    if chroma_host is None:
        chroma_host = os.environ.get('CHROMA_HOST', 'localhost')
    if chroma_port is None:
        chroma_port = int(os.environ.get('CHROMA_PORT', 8000))
    
    try:
        # Create Chroma client
        chroma_client = chromadb.HttpClient(
            host=chroma_host,
            port=chroma_port,
            ssl=chroma_ssl
        )
        
        # Create vector store with server configuration
        vector_store = Chroma(
            client=chroma_client,
            collection_name=chroma_collection_name,
            embedding_function=embeddings
        )
        
        # Verify connection by checking the collection
        vector_store.get()
    except Exception as e:
        # Log the connection error
        logging.warning(f"Could not connect to Chroma server at {chroma_host}:{chroma_port}. Falling back to in-memory vector store. Error: {e}")
        
        # Fallback to in-memory vector store
        vector_store = Chroma(
            collection_name=chroma_collection_name,
            embedding_function=embeddings
        )
    
    # Add documents if provided
    if documents:
        vector_store.add_documents(documents=documents)
    
    return vector_store