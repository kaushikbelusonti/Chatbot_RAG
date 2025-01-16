# Import required libraries
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.llms import OpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langsmith import Client
from langchain_community.callbacks import get_openai_callback
import streamlit as st
import os
import time

# Environment Setup
load_dotenv()
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

# Cache the preprocessing pipeline
@st.cache_resource
def preprocess_pipeline(pdf_path):
    # Load PDF documents and split pages
    loader = PyPDFLoader(pdf_path)
    pdf_pages = loader.load_and_split()

    # Split the documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    splits = text_splitter.split_documents(pdf_pages)

    # Create vector store
    vectorstore = FAISS.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    return retriever

# Cache the LLM
@st.cache_resource
def get_llm():
    return OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)

# Load retriever and LLM
retriever = preprocess_pipeline("Bhagavad-Gita_As_It_Is.pdf")
llm = get_llm()

# Define a function to format retrieved documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Define custom prompt
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# Create RAG chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Streamlit UI
st.title("Chatbot")
st.write("Enter your text below and click the 'Send' button to process it.")

user_input = st.text_input("Your text here:")

if st.button("Send"):
    if user_input:
        st.write("LLM Output")
        start_time = time.time()
        with get_openai_callback() as cb:
            response = rag_chain.invoke(str(user_input))
        end_time = time.time()
        st.write(response)
        st.write(f"Latency: {round(end_time - start_time, 2)} seconds")
        st.write(f"Tokens Used: {cb.total_tokens} [{cb.prompt_tokens} Prompt + {cb.completion_tokens} Completion]")
        st.write(f"Total Cost: ${round(cb.total_cost, 4)}")
    else:
        st.warning("Please enter some text before clicking Send.")
