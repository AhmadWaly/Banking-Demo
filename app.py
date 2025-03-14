import streamlit as st
import pandas as pd
import os
from pathlib import Path
import json
from openai import AzureOpenAI
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Haystack imports
from haystack import Document
from haystack import Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.converters.txt import TextFileToDocument
from haystack.components.preprocessors import DocumentCleaner
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever

# Set page configuration
st.set_page_config(
    page_title="QA Banking Chatbot Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if 'page' not in st.session_state:
    st.session_state.page = "upload"
if 'document_store' not in st.session_state:
    st.session_state.document_store = InMemoryDocumentStore()
if 'qa_content' not in st.session_state:
    st.session_state.qa_content = ""
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'pipeline_initialized' not in st.session_state:
    st.session_state.pipeline_initialized = False

# Create data directory if it doesn't exist
data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

# Function to initialize Haystack pipeline
def initialize_pipeline():
    document_store = InMemoryDocumentStore()
    p = Pipeline()
    p.add_component(instance=TextFileToDocument(), name="text_file_converter")
    p.add_component(instance=DocumentCleaner(), name="cleaner")
    p.add_component(instance=DocumentSplitter(split_by="sentence", split_length=4, split_overlap=1), name="splitter")
    p.add_component(instance=DocumentWriter(document_store=document_store), name="writer")
    p.connect("text_file_converter.documents", "cleaner.documents")
    p.connect("cleaner.documents", "splitter.documents")
    p.connect("splitter.documents", "writer.documents")
    
    path = "data"
    files = list(Path(path).glob("*.txt"))
    p.run({"text_file_converter": {"sources": files}})
    
    st.session_state.document_store = document_store
    st.session_state.pipeline_initialized = True
    return document_store

# Function to save QA content to a file
def save_qa_content(content):
    with open("data/qa_content.txt", "w") as f:
        f.write(content)
    st.session_state.qa_content = content

# Function to invoke the LLM
def invoke_llm(query):
    client = AzureOpenAI(
        azure_endpoint="https://segma-m7z0qwau-eastus2.openai.azure.com/openai/deployments/gpt-35-turbo-16k/chat/completions?api-version=2025-01-01-preview",
        api_key="FAiyazC61VKHyn8k8EtVhgLdgrXqu02Ce2Ujuqa5JiWgcBLSLNfEJQQJ99BCACHYHv6XJ3w3AAAAACOGU4xA",
        api_version="2024-12-01-preview"
    )
    
    # Enhanced prompt to ensure strict adherence to provided context
    prompt = """You are a QA customer service bot for a bank.\
          Your responsibility is to answer queries by the bank's clients based on a given context.\
              In order to perfect your goal, follow these steps: - Your answer should be aligned with the given context. \
                - If the question was part of the given context, please remove it. 
                    - Don't add any information or assumptions that are not part of the given context. 
                    - If the given context does not align with the question, please don't answer. 
                    Example 1: User query: "I need to apply for a loan"
                      context: "Q) What are the benefits of online banking? 
                      Answer: Online banking allows you to access your accounts 24/7, view transaction history, transfer funds, pay bills, and more." Bot I don't have engouhh info.
                        Example 2: User query: "What are the requirements for opening a checking account?" 
                        context: "Q) What are the requirements for opening a checking account?
                          Answer: Requirements for opening a checking account may vary depending on the bank. Generally, you will need to provide identification and proof of residency." Bot Requirements for opening a checking account may vary depending on the bank. Generally, you will need to provide identification and proof of residency. ."""
    
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": query}
    ]
    
    # Log the full query being sent to the LLM
    logger.info(f"Full query sent to LLM: {query}")
    
    completion = client.chat.completions.create(
        model="gpt-35-turbo-16k",
        messages=messages,
        max_tokens=1000,  # Increased token limit for more complete answers
        temperature=0,
        top_p=0.9,
        frequency_penalty=0.9,
        presence_penalty=0,
        stop=None,
        stream=False
    )
    
    res = completion.to_json()
    res = json.loads(res)
    return res["choices"][0]["message"]["content"]

# Function to process user query
def process_query(query):
    if not st.session_state.pipeline_initialized:
        initialize_pipeline()
    
    retriever = InMemoryBM25Retriever(document_store=st.session_state.document_store)
    results = retriever.run(query=query, top_k=1)  # Increased to top 3 for better context
    
    if not results["documents"]:
        return "I don't have information about that in my knowledge base."
    
    # Combine content from retrieved documents for better context
    context = "\n".join([doc.content for doc in results["documents"]])
    full_query = f"query: {query}\ncontext: {context}"
    
    # Log the full query to console/logs
    print("=" * 50)
    print(f"FULL QUERY LOG:")
    print(full_query)
    print("=" * 50)
    
    response = invoke_llm(full_query)
    
    # Add to chat history
    st.session_state.chat_history.append({"user": query, "bot": response})
    return response

# Function to switch to chat page
def switch_to_chat():
    if st.session_state.qa_content:
        save_qa_content(st.session_state.qa_content)
        initialize_pipeline()
        st.session_state.page = "chat"
    else:
        st.error("Please enter some QA content before proceeding to chat.")

# Navigation sidebar
st.sidebar.title("Navigation")
if st.sidebar.button("Upload Banking QA Content"):
    st.session_state.page = "upload"
if st.sidebar.button("Chat with Banking Agent"):
    if os.path.exists("data/qa_content.txt"):
        if not st.session_state.pipeline_initialized:
            initialize_pipeline()
        st.session_state.page = "chat"
    else:
        st.sidebar.error("Please upload Banking QA content first.")

st.sidebar.divider()
st.sidebar.info("This application demonstrates a Banking QA Agent that answers questions based on provided content.")

# Display appropriate page based on session state
if st.session_state.page == "upload":
    st.title("📚 Upload QA Content")
    
    # Load existing content if available
    if os.path.exists("data/qa_content.txt"):
        with open("data/qa_content.txt", "r") as f:
            st.session_state.qa_content = f.read()
    
    st.write("Enter your Question and Answer content below. Format your content with clear questions and answers for best results.")
    
    # Text area for QA content input
    qa_content = st.text_area(
        "Question and Answer Content",
        value=st.session_state.qa_content,
        height=400,
        help="Enter your QA content here. Each question should be followed by its answer.",
    )
    
    # Example format
    with st.expander("Example Format"):
        st.code("""Q: How do I make a bill payment?
A: To make a bill payment, log into your account, go to the 'Payments' section, select the bill you want to pay, enter the amount, and click 'Submit'.

Q: What are the business hours?
A: Our business hours are Monday to Friday from 9 AM to 5 PM, and Saturday from 10 AM to 2 PM. We are closed on Sundays and holidays.

Q: How can I reset my password?
A: To reset your password, click on the 'Forgot Password' link on the login page, enter your email address, and follow the instructions sent to your email.""")
    
    # Save button
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Save QA Content"):
            save_qa_content(qa_content)
            st.success("QA content saved successfully!")
    
    with col2:
        if st.button("Proceed to Chat"):
            st.session_state.qa_content = qa_content
            switch_to_chat()

elif st.session_state.page == "chat":
    st.title("🤖 Chat with Banking Agent")
    
    # Display explanation of bot limitations
    st.info("This Agent will only answer questions based on the QA content you provided. If the information isn't in the knowledge base, it will let you know.")
    
    # Display chat history
    for chat in st.session_state.chat_history:
        st.markdown(f"**You:** {chat['user']}")
        st.markdown(f"**Bot:** {chat['bot']}")
        st.divider()
    
    # Chat input
    user_query = st.chat_input("Ask a question...")
    if user_query:
        with st.spinner("Processing your query..."):
            response = process_query(user_query)
        
        # Force a rerun to update the chat history display
        st.rerun()
    
    # Option to clear chat history
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

    # Display a debug section for administrators
    with st.expander("Debug Information"):
        st.write("This section shows retrieval details for administrators.")
        if st.button("Show document store statistics"):
            doc_count = len(st.session_state.document_store.filter_documents({}))
            st.write(f"Number of documents in store: {doc_count}")