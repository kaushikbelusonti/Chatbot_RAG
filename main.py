
from ragchain import RAGChain

# Create a RAG chain instance
chain = RAGChain(model_name='gpt-3.5-turbo', 
    temperature=0,
    chunk_size=100,
    chunk_overlap=10,
    document_folder='BhagvadGita_Reduced.pdf')

# Configure environment
chain.configure_environment()
# Initialize RAG Chain components
chain._initialize_components()
# Load and Process documents
docs = chain.load_documents()
chain.process_documents(documents=docs)
# Configure RAG Chain
chain.setup_rag_chain(k=3)
# Invoke RAG Chain
Result = chain.query("Who is Krishna")
print(Result)