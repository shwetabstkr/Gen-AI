import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain.chains.question_answering import load_qa_chain
 
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
 
st.title("RAG Chatbot with FAISS and LLaMA")
st.write("Upload a PDF and ask questions based on its content.")
 
uploaded_file = st.file_uploader("Upload your PDF file", type="pdf")
if uploaded_file:
    os.makedirs("uploaded", exist_ok=True)
    pdf_path = os.path.join("uploaded", uploaded_file.name)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    text = extract_text_from_pdf(pdf_path)
    if text:
        st.info("Creating FAISS vector store...")
        create_faiss_vector_store(text)
        st.info("Initializing chatbot...")
        qa_chain = build_qa_chain()
        st.session_state['qa_chain'] = qa_chain
        st.success("Chatbot is ready!")
 
if 'qa_chain' in st.session_state:
    question = st.text_input("Ask a question about the uploaded PDF:")
    if question:
        st.info("Querying the document...")
        answer = st.session_state['qa_chain'].run(question)
        st.success(f"Answer: {answer}")