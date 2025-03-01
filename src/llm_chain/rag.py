from typing_extensions import List, TypedDict
from langchain import hub
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph

class State(TypedDict):
    """
    State for the RAG application.
    
    Attributes:
        question (str): The user's question
        context (List[Document]): Retrieved context documents
        answer (str): Generated answer
    """
    question: str
    context: List[Document]
    answer: str

def retrieve(state: State, vector_store):
    """
    Retrieve relevant documents for the given question.
    
    Args:
        state (State): Current state of the application
        vector_store: Vector store to search
    
    Returns:
        Dict with retrieved context
    """
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

def generate(state: State, llm, prompt):
    """
    Generate an answer based on the retrieved context.
    
    Args:
        state (State): Current state of the application
        llm: Language model to generate answer
        prompt: Prompt template
    
    Returns:
        Dict with generated answer
    """
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

def create_rag_graph(llm, vector_store):
    """
    Create a RAG application graph.
    
    Args:
        llm: Language model
        vector_store: Vector store for retrieval
    
    Returns:
        Compiled graph
    """
    # Pull the prompt template
    prompt = hub.pull("rlm/rag-prompt")
    
    # Create graph builder
    graph_builder = StateGraph(State)
    
    # Define graph steps
    graph_builder.add_node("retrieve", lambda state: retrieve(state, vector_store))
    graph_builder.add_node("generate", lambda state: generate(state, llm, prompt))
    
    # Define graph edges
    graph_builder.add_edge(START, "retrieve")
    graph_builder.add_edge("retrieve", "generate")
    
    # Compile the graph
    return graph_builder.compile()