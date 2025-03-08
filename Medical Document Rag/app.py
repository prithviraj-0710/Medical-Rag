import streamlit as st
import os
import time
from langchain_groq import ChatGroq
from transformers import pipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv


load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")


llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")


prompt = ChatPromptTemplate.from_template(
     """
    Answer the question using the provided context **if relevant**. 
    If the context is not useful, provide a general response based on your knowledge.

    <context>
    {context}
    </context>

    Question: {input}

    Provide a well-reasoned response while staying aligned with the context when applicable.
    """
)

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text(text):
    """Summarizes the given text using a pre-trained transformer model."""
    summary = summarizer(text, max_length=150, min_length=50, do_sample=False)
    return summary[0]['summary_text']


def create_vector_embedding(uploaded_file=None):
    """Creates vector embeddings for the provided document (if any) or default dataset."""
    
    if "vectors" not in st.session_state:
        with st.spinner("🔄 Processing document embeddings... Please wait."):

            st.session_state.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

            # Check if a user uploaded a medical history PDF
            if uploaded_file:
                temp_path = "uploaded_medical.pdf"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                st.session_state.loader = PyPDFLoader(temp_path)  
            else:
                st.session_state.loader = PyPDFDirectoryLoader("pharma_data")

            st.session_state.docs = st.session_state.loader.load()

            if not st.session_state.docs:
                st.error("⚠️ No documents found! Upload a file or check dataset.")
                return  

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=30)
            st.session_state.final_documents = text_splitter.split_documents(st.session_state.docs)

            st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
            st.success("✅ Vector Database is ready!")



st.markdown(
    "<h1 style='text-align: center; color: #4CAF50;'>RAG-Powered Medical Q&A</h1>", 
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center; font-size:18px;'>Ask any question based on the Medical Part D dataset.</p>", 
    unsafe_allow_html=True
)

# PDF Upload Section
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
if uploaded_file:
    st.success("✅ File uploaded! Click 'Generate Document Embeddings' to process.")



if st.button("🔍 Generate Document Embeddings"):
    create_vector_embedding()


# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display previous messages
for message in st.session_state.chat_history:
    role, content = message
    with st.chat_message(role):
        st.write(content)

# Chat input field
user_prompt = st.chat_input("Ask something...")

if user_prompt:
    if "vectors" not in st.session_state:
        st.error("⚠️ Please generate document embeddings first.")
    else:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        with st.chat_message("user"):
            st.write(user_prompt)

        start_time = time.process_time()
        retrieved_docs = retriever.get_relevant_documents(user_prompt)

# Check if relevant documents were found
        if retrieved_docs and any(doc.page_content.strip() for doc in retrieved_docs):
            response = retrieval_chain.invoke({'input': user_prompt})
            answer = response.get('answer', "⚠️ No valid response found.")

            if len(answer) > 200:  # Only summarize if necessary
                answer = summarize_text(answer)
        else:
            st.warning("⚠️ No highly relevant documents found. Providing a general AI response.")
            answer = llm.invoke(user_prompt)
            answer = answer if isinstance(answer, str) else answer.content  # Ensure response is a string

        response_time = time.process_time() - start_time

        with st.chat_message("assistant"):
            st.write(answer)

        # Store chat history
        st.session_state.chat_history.append(("user", user_prompt))
        st.session_state.chat_history.append(("assistant", answer))
