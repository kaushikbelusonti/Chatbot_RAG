import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='tqdm')
import logging # Python build-in logging module. No need to install any external library
from logging.handlers import RotatingFileHandler
import os
# Set tokenizers parallelism before importing any HuggingFace components
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dotenv import load_dotenv
from typing import Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from logger_config import LoggerConfig
from pathlib import Path

# Initialize logging
logger_config = LoggerConfig(log_dir="logs/ragchain")
logger = logger_config.get_logger("rag_chain")
error_logger = logger_config.get_error_logger("rag_chain")

class RAGChain:
    """
    Retrieval Augmented Generation (RAG) Chain for document processing and question answering.
    Supports both OpenAI and HuggingFace models for embeddings and LLM capabilities.
    """
    def __init__(self,
        model_name: str = 'gpt-3.5-turbo', 
        temperature: float = 0,
        use_openai_embeddings = False,
        chunk_size: int = 100,
        chunk_overlap: int = 10,
        document_folder: Optional[str] = None):
        """
        Initialize the RAG Chain with specified parameters.
        
        Args:
            model_name (str): Name of the LLM model to use
            temperature (float): Temperature for LLM responses
            chunk_size (int): Size of text chunks for processing
            chunk_overlap (int): Overlap between chunks
            document_folder (str, optional): Folder containing documents to process
            use_openai_embeddings (bool): Whether to use OpenAI embeddings instead of HuggingFace
            embedding_model (str): HuggingFace model to use for embeddings
        """
        
        self.model_name = model_name
        self.temperature = temperature
        self.use_openai_embeddings = use_openai_embeddings
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.document_folder = document_folder
        
        logger.info(f"Initializing RAGChain with model: {model_name}")
        
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
        try:
            load_dotenv()
            required_vars = ['LANGCHAIN_TRACING', 'LANGCHAIN_ENDPOINT', 'LANGCHAIN_API_KEY', 'OPENAI_API_KEY']
            
            missing_vars = [var for var in required_vars if not os.getenv(var)]
            
            if missing_vars:
                raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
            
            logger.info("Environment Configured Successfully")
        except Exception as e:
            error_logger.error(f"Error Configuring Environment: {str(e)}")
            raise # Re-raises the exception to prevent further execution            
        
    def _initialize_components(self):
        
        try:
            # Initialize LLM
            self.llm = ChatOpenAI(model_name=self.model_name, temperature=self.temperature)
            
            # Initialize embeddings based on configurations
            if self.use_openai_embeddings:
                self.embeddings = OpenAIEmbeddings()
                logger.info("Using OpenAI embeddings")
            else:
                self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                logger.info("Using HuggingFace Embeddings")
                
            # Initialize text splitter
            self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
            logger.info("Components initialized successfully")
        
        except Exception as e:
            error_logger.error(f"Error initializing components: {str(e)}") 
            raise

    def _load_and_process_documents(self):
        
        try:
            if not self.document_folder:
                raise ValueError("Document folder not specified")
            
            folder_path = Path(self.document_folder)
            if not folder_path.exists():
                raise FileNotFoundError(f"Document folder not found: {self.document_folder}")
            
            """Load and process all documents from the folder and its sub-folders"""
            documents = []
            pdf_count = 0
            
            # Traverse the document_folder recursively and load all documents
            for dirpath, _, filenames in os.walk(self.document_folder):
                for filename in filenames:
                    try:
                        # Ensure the file is a PDF or a document type you want to process
                        if filename.endswith(".pdf"):  # You can add more extensions if needed
                            filepath = os.path.join(dirpath, filename)
                            logger.info(f"Loading document: {filepath}")  
                            loader = PyMuPDFLoader(filepath)
                            documents.extend(loader.load_and_split())  # Add loaded documents to the list
                            pdf_count += 1
                    except Exception as e:
                        error_logger.error(f"Error processing {filepath}: {str(e)}")
                        continue
            
            if pdf_count == 0:
                raise ValueError(f"No PDF files found")
            
            # Process documents
            self.chunks = self.text_splitter.split_documents(documents)
            logger.info(f"Created {len(self.chunks)} chunks from {pdf_count} documents")
            
            # Create vector store
            self.vector_store = FAISS.from_documents(documents=self.chunks, embedding=self.embeddings)
            logger.info(f"Vector store created successfully")
            
        except Exception as e:
            error_logger.error(f"Error in document processing: {str(e)}")
            raise
        
    def get_chunk_count(self):
        return len(self.chunks) if self.chunks else 0
        
    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)
        
    def setup_rag_chain(self, k: int = 3):
        
        try:
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
            logger.info(f"RAG Chain setup completed")
            return self.rag_chain
            
        except Exception as e:
            error_logger.error(f"Error setting up RAG chain: {str(e)}")
            raise
            
    def query(self, question) -> str:
        try:
            logger.info(f"Processing query: {question}")
            response = self.rag_chain.invoke(question)
            logger.info("Query processed successfully")
            return self.rag_chain.invoke(question)
        except Exception as e:
            error_logger.error(f"Error processing query: {str(e)}")
            raise