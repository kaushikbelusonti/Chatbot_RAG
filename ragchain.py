from dotenv import load_dotenv
from typing import Optional
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

class RAGChain:
    def __init__(self,
        model_name: str = 'gpt-3.5-turbo', 
        temperature: float = 0,
        chunk_size: int = 100,
        chunk_overlap: int = 10,
        document_folder: Optional[str] = None):
        
        self.model_name = model_name
        self.temperature = temperature
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.document_folder = document_folder
        
        # Configure environment
        self.configure_environment()
        
        # Initialize components
        self.llm = None
        self.embeddings = None
        self.text_splitter = None
        self.vector_store = None
        
        self._initialize_components()
        self._load_and_process_documents()
        self.setup_rag_chain()
        
    def configure_environment(self):
        load_dotenv()
        os.environ['LANGCHAIN_TRACING_V2'] = os.getenv('LANGCHAIN_TRACING')
        os.environ['LANGCHAIN_ENDPOINT'] = os.getenv('LANGCHAIN_ENDPOINT')
        os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
        os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
        
    def _initialize_components(self):
        """Initialize LangChain components"""
        self.llm = ChatOpenAI(model_name=self.model_name, temperature=self.temperature)
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)

    def _load_and_process_documents(self):
        """Load and process all documents from the folder and its sub-folders"""
        documents = []
        
        # Traverse the document_folder recursively and load all documents
        for dirpath, _, filenames in os.walk(self.document_folder):
            for filename in filenames:
                # Ensure the file is a PDF or a document type you want to process
                if filename.endswith(".pdf"):  # You can add more extensions if needed
                    filepath = os.path.join(dirpath, filename)
                    #print(f"Loading document: {filepath}")  # Debugging statement
                    loader = PyMuPDFLoader(filepath)
                    documents.extend(loader.load_and_split())  # Add loaded documents to the list

        self.chunks = self.text_splitter.split_documents(documents)
        self.vector_store = FAISS.from_documents(
            documents=self.chunks,
            embedding=self.embeddings)
    
    def get_chunk_count(self):
        return len(self.chunks) if self.chunks else 0
        
    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)
        
    def setup_rag_chain(self, k: int = 3):
        retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
        # Define custom prompt
        template = """Answer the question based only on the following context:
        {context}

        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)
        self.rag_chain = (
            {"context": retriever | self.format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return self.rag_chain
    
    def query(self, question) -> str:
        return self.rag_chain.invoke(question)