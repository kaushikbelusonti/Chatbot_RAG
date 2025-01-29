from dotenv import load_dotenv
from typing import Optional
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
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
        # Configure environment
        self.configure_environment()
        
        self.model_name = model_name
        self.temperature = temperature
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.document_folder = document_folder
        
        # Initialize components
        self.llm = None
        self.embeddings = None
        self.textsplitter = None
        self.vectorstore = None
        
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
        self.textsplitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        
    def load_documents(self):
        """Load PDF document from a directory"""
        # Load only PDF documents
        loader = PyPDFLoader(self.document_folder)
        # Split the pdf into pages
        documents = loader.load_and_split()
        return documents
    
    def process_documents(self, documents) -> None:
        chunks = self.textsplitter.split_documents(documents)
        self.vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings)
    
    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)
        
    def setup_rag_chain(self, k: int = 3):
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
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