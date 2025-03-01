from llm_chain.llm import get_llm, get_embeddings
from llm_chain.vectorstore import load_documents, chunk_documents, create_vector_store
from llm_chain.rag import create_rag_graph, State
from langchain import hub
from typing_extensions import List
import os

def main():
    # Initialize LLM and Embeddings
    llm = get_llm()
    embeddings = get_embeddings()

    # Optional: Configure Chroma server (can also use environment variables)
    chroma_host = os.environ.get('CHROMA_HOST', 'localhost')
    chroma_port = os.environ.get('CHROMA_PORT', 8000)

    # Load and chunk contents of the blog
    docs = load_documents(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",)
    )
    all_splits = chunk_documents(docs)

    # Create vector store with optional server configuration
    vector_store = create_vector_store(
        embeddings, 
        documents=all_splits,
        chroma_host=chroma_host,
        chroma_port=int(chroma_port),
        chroma_collection_name='test_collection'
    )

    # Pull the prompt for demonstration
    prompt = hub.pull("rlm/rag-prompt")

    # Demonstrate prompt invocation
    example_messages = prompt.invoke(
        {"context": "(context goes here)", "question": "(question goes here)"}
    ).to_messages()

    # Verify prompt message count
    assert len(example_messages) == 1
    print("Prompt message content:", example_messages[0].content)

    # Create and compile the RAG graph
    graph = create_rag_graph(llm, vector_store)

    # Invoke the graph with a specific question
    result = graph.invoke({"question": "What is Task Decomposition?"})

    # Print context and answer
    print(f'\nContext: {result["context"]}\n')
    print(f'Answer: {result["answer"]}')

if __name__ == "__main__":
    main()
