import streamlit as st
from ragchain import RAGChain
import time
from langchain_community.callbacks.manager import get_openai_callback  # Import OpenAI callback
#from langchain.callbacks import get_openai_callback  # Import OpenAI callback


# Set Streamlit Page Configuration
st.set_page_config(page_title="RAGChain QA", layout="wide")

# Title of the app
st.title("📖 RAGChain: Ask Questions from Documents")

# Initialize RAGChain instance only once
@st.cache_resource
def load_rag_chain():
    return RAGChain(
        use_openai_llm = False,
        llm_temperature=0,
        use_openai_embeddings=False,
        chunk_size=100,
        chunk_overlap=10,
        document_folder='/Users/kaushikbelusonti/Chatbot_RAG/Documents')

# Load the RAG Chain instance
chain = load_rag_chain()

# User input for query
question = st.text_input("🔍 Ask a question:", "")

# Process the query when the user presses Enter or clicks a button
if question:
    with st.spinner("Thinking... 💭"):
        start_time = time.time() # Start measuring time
        
        # Query the RAG Chain
        with get_openai_callback() as cb:
            response = chain.query(question)  # Invoke the RAG Chain
        
        stop_time = time.time() # Stop measuring time
        
        # Latency
        latency = round(time.time() - start_time, 2)
        
    # Display results
    st.subheader("📝 Answer:")
    st.write(response)
    
    # Display performance metric
    st.subheader("📊 Performance Metrics:")
    st.write(f"Latency: {round(stop_time - start_time, 2)} seconds")
    st.write(f"Tokens Used: {cb.total_tokens} [{cb.prompt_tokens} Prompt + {cb.completion_tokens} Completion]")
    st.write(f"Total Cost: ${round(cb.total_cost, 4)}")