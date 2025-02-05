# Chatbot with LangChain and Streamlit
This project creates a chatbot application using Streamlit, LangChain, and OpenAI. The chatbot retrieves relevant information from a PDF document (e.g., the Bhagavad Gita), splits the document into chunks, generates vector embeddings, and uses these embeddings for context-based question answering.

#### Features
- PDF Document Loading: Load and split PDF documents into smaller chunks.
- Text Preprocessing: Split the document text into manageable chunks using LangChain's RecursiveCharacterTextSplitter.
- Vector Store: Use FAISS (Facebook AI Similarity Search) to store vector embeddings of the document chunks.
- Retriever: Efficient document retrieval based on similarity to the user's query.
- Language Model: Uses OpenAI's GPT-3.5-turbo for generating responses.
- Latency and Token Tracking: Track response latency, tokens used, and cost per request using LangChain and OpenAI callback.

#### Requirements

###### Clone the repository:
```bash
git clone https://github.com/kaushikbelusonti/Chatbot_RAG.git
cd Chatbot_RAG
```

###### Install the dependencies using pip: 
```bash
pip install -r requirements.txt
```

#### Setup
**API Keys:** Set up your environment variables in a .env file for OPENAI_API_KEY and LANGCHAIN_API_KEY:
```bash
OPENAI_API_KEY=your_openai_api_key
LANGCHAIN_API_KEY=your_langchain_api_key
```

#### Application Structure

###### Preprocessing Pipeline:
  - Loads the PDF, splits the document into chunks, and creates a retriever for document retrieval. 
###### LLM Setup:
  - Uses OpenAI's GPT model to generate responses.
###### Custom Prompt:
  - A template-based prompt to structure the user query and the retrieved context.
###### Streamlit Interface:
  - A simple UI with a text input field for the user to submit queries.
  - Displays the response from the model, along with latency, tokens used, and the cost of the API call.

#### Running the Application
To run the chatbot application, use the following command:
```bash
streamlit run main.py
```
This will launch a Streamlit app in your browser. You can enter a query, and the chatbot will respond based on the context from the loaded PDF document.

#### Cost and Latency
The application will display:
- Latency: Time taken for the model to generate a response.
- Tokens Used: Tokens consumed in the request (both prompt and completion).
- Cost: The cost for the API request based on the token usage.



