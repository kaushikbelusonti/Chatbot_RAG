import streamlit as st
from ragchain import RAGChain

# Set Streamlit Page Configuration
st.set_page_config(page_title="RAGChain QA", layout="wide")

# Title of the app
st.title("📖 RAGChain: Ask Questions from Documents")

# Initialize RAGChain instance only once
@st.cache_resource
def load_rag_chain():
    return RAGChain(model_name='gpt-3.5-turbo', 
        temperature=0,
        chunk_size=100,
        chunk_overlap=10,
        document_folder='BhagvadGita_Reduced.pdf')

# Load the RAG Chain instance
chain = load_rag_chain()

# User input for query
question = st.text_input("🔍 Ask a question:", "")

# Process the query when the user presses Enter or clicks a button
if question:
    with st.spinner("Thinking... 💭"):
        result = chain.query(question)
    st.subheader("📝 Answer:")
    st.write(result)