import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain.chains.question_answering import load_qa_chain
import time


# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            txt = page.extract_text()
            if txt:
                text += txt
        return text
    except Exception as e:
        st.error(f"Failed to extract text: {e}")
        return ""

# Function to create FAISS vector store
def create_faiss_vector_store(text, path="faiss_index"):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local(path)

# Load FAISS vector store
def load_faiss_vector_store(path="faiss_index"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    return vector_store

# Build QA Chain
def build_qa_chain(vector_store_path="faiss_index"):
    vector_store = load_faiss_vector_store(vector_store_path)
    retriever = vector_store.as_retriever()
    llm = Ollama(model="llama3", base_url="http://ollama:11434")
    qa_chain = load_qa_chain(llm, chain_type="stuff")
    qa_chain = RetrievalQA(retriever=retriever, combine_documents_chain=qa_chain)
    return qa_chain

# Page configuration
st.set_page_config(
    page_title="🧠 AI Document Assistant", 
    page_icon="🤖", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .upload-section {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin-bottom: 2rem;
    }
    
    .chat-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
    }
    
    .status-box {
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid;
    }
    
    .status-success {
        background-color: #d4edda;
        border-color: #28a745;
        color: #155724;
    }
    
    .status-info {
        background-color: #d1ecf1;
        border-color: #17a2b8;
        color: #0c5460;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'qa_chain' not in st.session_state:
    st.session_state['qa_chain'] = None
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
if 'pdf_processed' not in st.session_state:
    st.session_state['pdf_processed'] = False

# Sidebar
with st.sidebar:
    st.markdown("## 🛠️ Assistant Controls")
    
    # System status
    st.markdown("### 📊 System Status")
    if st.session_state['pdf_processed']:
        st.success("✅ Document Loaded")
        st.success("✅ AI Assistant Ready")
    else:
        st.warning("⏳ Waiting for document")
        st.info("ℹ️ Upload a PDF to begin")
    
    st.markdown("---")
    
    # Statistics
    st.markdown("### 📈 Session Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Questions", len(st.session_state['chat_history']))
    with col2:
        st.metric("Status", "Active" if st.session_state['qa_chain'] else "Idle")
    
    st.markdown("---")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History", type="secondary"):
        st.session_state['chat_history'] = []
        st.rerun()
    
    st.markdown("---")
    
    # Information
    st.markdown("### ℹ️ How it works")
    st.info("""
    1. Upload your PDF document
    2. Wait for processing to complete
    3. Ask questions about the content
    4. Get AI-powered answers
    """)

# Main content area
st.markdown("""
<div class="main-header">
    <h1>🧠 AI Document Assistant</h1>
    <p>Upload any PDF document and chat with an AI that understands its content</p>
</div>
""", unsafe_allow_html=True)

# Document upload section
st.markdown("## 📄 Document Upload")

with st.container():
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose your PDF file", 
            type="pdf",
            help="Upload a PDF document to analyze with AI",
            label_visibility="collapsed"
        )
    
    with col2:
        if uploaded_file:
            st.markdown("**File Details:**")
            st.write(f"📄 {uploaded_file.name}")
            st.write(f"📏 {uploaded_file.size / 1024:.1f} KB")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Process uploaded file
if uploaded_file and not st.session_state['pdf_processed']:
    with st.spinner("🔄 Processing your document..."):
        # Create progress bar
        progress_bar = st.progress(0)
        
        # Save uploaded file
        os.makedirs("uploaded", exist_ok=True)
        pdf_path = os.path.join("uploaded", uploaded_file.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        progress_bar.progress(25)
        
        # Extract text
        st.info("🔍 Extracting text from PDF...")
        text = extract_text_from_pdf(pdf_path)
        progress_bar.progress(50)
        
        if text:
            # Create vector store
            st.info("🧮 Creating knowledge embeddings...")
            create_faiss_vector_store(text)
            progress_bar.progress(75)
            
            # Initialize chatbot
            st.info("🤖 Initializing AI assistant...")
            qa_chain = build_qa_chain()
            st.session_state['qa_chain'] = qa_chain
            st.session_state['pdf_processed'] = True
            progress_bar.progress(100)
            
            time.sleep(0.5)  # Small delay for UX
            st.success("🎉 Document processed successfully! You can now ask questions.")
            st.rerun()

# Chat interface
if st.session_state['qa_chain']:
    st.markdown("## 💬 Chat with your Document")
    
    # Display chat history
    if st.session_state['chat_history']:
        st.markdown("### 📜 Conversation History")
        for i, (q, a) in enumerate(st.session_state['chat_history']):
            with st.expander(f"💭 Question {i+1}: {q[:50]}...", expanded=(i == len(st.session_state['chat_history'])-1)):
                st.markdown(f"**🙋 You asked:** {q}")
                st.markdown(f"**🤖 AI responded:** {a}")
    
    # Question input
    st.markdown("### ❓ Ask a New Question")
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        question = st.text_input(
            "What would you like to know about the document?",
            placeholder="e.g., What is the main topic of this document?",
            label_visibility="collapsed"
        )
    
    with col2:
        ask_button = st.button("🚀 Ask", type="primary", use_container_width=True)
    
    # Process question
    if (question and ask_button) or (question and st.session_state.get('last_question') != question):
        st.session_state['last_question'] = question
        
        with st.spinner("🤔 AI is thinking..."):
            try:
                # Simulate thinking time for better UX
                progress = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress.progress(i + 1)
                
                answer = st.session_state['qa_chain'].run(question)
                
                # Add to chat history
                st.session_state['chat_history'].append((question, answer))
                
                # Display answer with enhanced styling
                st.markdown("### 🎯 Latest Response")
                st.markdown(f"""
                <div class="chat-container">
                    <h4>🙋 Your Question:</h4>
                    <p>{question}</p>
                    <h4>🤖 AI Response:</h4>
                    <p>{answer}</p>
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
    
    # Suggested questions
    if not st.session_state['chat_history']:
        st.markdown("### 💡 Suggested Questions")
        suggestions = [
            "What is the main topic of this document?",
            "Can you summarize the key points?",
            "What are the most important findings?",
            "Are there any specific recommendations mentioned?"
        ]
        
        cols = st.columns(2)
        for i, suggestion in enumerate(suggestions):
            with cols[i % 2]:
                if st.button(f"💭 {suggestion}", key=f"suggestion_{i}"):
                    st.session_state['suggested_question'] = suggestion
                    st.rerun()

else:
    # Welcome screen
    st.markdown("## 🚀 Getting Started")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>📤 Upload</h3>
            <p>Upload any PDF document you want to analyze</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🤖 Process</h3>
            <p>AI creates embeddings and prepares for questions</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>💬 Chat</h3>
            <p>Ask questions and get intelligent answers</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit, LangChain, and Llama AI")
