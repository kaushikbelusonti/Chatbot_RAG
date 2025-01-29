
from ragchain import RAGChain

# Create a RAG chain instance
chain = RAGChain(model_name='gpt-3.5-turbo', 
    temperature=0,
    chunk_size=100,
    chunk_overlap=10,
    document_folder='BhagvadGita_Reduced.pdf')

Result = chain.query("Who is Krishna")
print(Result)