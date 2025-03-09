import streamlit as st
import google.generativeai as genai
import pdfplumber
import faiss
import numpy as np
from dotenv import load_dotenv
import os

# Set page configuration FIRST
st.set_page_config(
    page_title="PDF Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Gemini API key from .env
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Clear chat history on app restart
if "messages" in st.session_state:
    st.session_state.messages = []

# Initialize Faiss vector store
dimension = 768  # Gemini embedding dimension
index = faiss.IndexFlatL2(dimension)
store = {}  # {chunk_id: text}

# Custom CSS for professional styling
st.markdown("""
    <style>
    /* General Styling */
    .stApp {
        background-color: #1e1e2f; /* Dark blue-gray background */
        color: #ffffff; /* White text */
    }
    h1 {
        color: #ff6f61; /* Coral red title */
        font-family: 'Helvetica', sans-serif;
    }
    .sidebar .sidebar-content {
        background-color: #2c2c40; /* Darker sidebar */
    }
    .stButton>button {
        background-color: #ff6f61; /* Coral red button */
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
        border: none;
    }
    .stChatMessage.user {
        background-color: #3a3a52; /* Dark gray for user messages */
        border-radius: 10px;
        padding: 10px;
    }
    .stChatMessage.assistant {
        background-color: #4a4a62; /* Slightly lighter gray for assistant messages */
        border-radius: 10px;
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)

def extract_text_from_pdf(file):
    with pdfplumber.open(file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

def split_text(text, chunk_size=300, overlap=50):
    chunks = []
    words = text.split()
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def generate_embeddings(chunks):
    embeddings = []
    for chunk in chunks:
        response = genai.embed_content(
            model="models/embedding-001",
            content=chunk,
            task_type="retrieval_document"
        )
        embeddings.append(response["embedding"])
    return np.array(embeddings)

def retrieve_relevant_chunks(query, top_k=3):
    query_embedding = genai.embed_content(
        model="models/embedding-001",
        content=query,
        task_type="retrieval_query"
    )["embedding"]
    
    _, indices = index.search(np.array([query_embedding]), top_k)
    relevant_chunks = [store[i] for i in indices[0]]
    return relevant_chunks

def generate_answer(query, context):
    prompt = f"""
    📚 You are an expert assistant helping a student study for exams.
    Answer the user's question based on the provided context below.
    If the context does not contain enough information, say "I don't know."

    Context: {context}

    Question: {query}

    Answer:
    """
    model = genai.GenerativeModel('gemini-1.5-pro')
    response = model.generate_content(prompt)
    return response.text

# Header Section
col1, col2 = st.columns([1, 3])
with col1:
    st.markdown("")  # Simple emoji for PDF instead of an image
with col2:
    st.title("📚 IntelliPDF: Your AI-Powered Study Companion")
    st.markdown("Ask questions about your PDF files using advanced AI models.")

# Sidebar for additional features
with st.sidebar:
    st.header("📝 About")
    st.write("IntelliPDF is a smart chatbot that helps you interact with your PDF notes.")
    st.markdown("""
    **Features:**
    - 📁 Upload PDF files
    - 💬 Ask questions about the content
    - 🔍 Retrieve relevant information
    - 🎯 Fine-tuned prompts for accuracy
    """)

# Main Content
uploaded_file = st.file_uploader("📁 Upload your PDF file", type="pdf", help="Upload a PDF file to get started.")

if uploaded_file:
    with st.spinner("⏳ Processing PDF..."):
        # Extract text and create embeddings
        text = extract_text_from_pdf(uploaded_file)
        chunks = split_text(text)
        embeddings = generate_embeddings(chunks)
        
        # Populate Faiss index and store
        index.add(embeddings)
        for i, chunk in enumerate(chunks):
            store[i] = chunk

    st.success("✅ PDF processed successfully! Ask me anything about it.")

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "🤖"):
        st.markdown(message["content"])

if query := st.chat_input("💬 Ask a question about the PDF"):
    # Add user message to session state
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user", avatar="👤"):
        st.markdown(query)

    # Generate bot response
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("🔍 Searching the PDF..."):
            relevant_chunks = retrieve_relevant_chunks(query)
            context = "\n".join(relevant_chunks)
        
        with st.spinner("📝 Generating answer..."):
            answer = generate_answer(query, context)
            st.markdown(answer)
        
        # Add bot message to session state
        st.session_state.messages.append({"role": "assistant", "content": answer})