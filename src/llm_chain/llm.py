from langchain.chat_models import init_chat_model
from langchain_google_vertexai import VertexAIEmbeddings

def get_llm(model_name="gemini-2.0-flash-001", model_provider="google_vertexai"):
    """
    Initialize and return a language model.
    
    Args:
        model_name (str): Name of the model to use
        model_provider (str): Provider of the model
    
    Returns:
        Initialized language model
    """
    return init_chat_model(model_name, model_provider=model_provider)

def get_embeddings(model="text-embedding-004"):
    """
    Initialize and return embeddings.
    
    Args:
        model (str): Name of the embedding model
    
    Returns:
        Initialized embeddings
    """
    return VertexAIEmbeddings(model=model)