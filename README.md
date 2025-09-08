# RAG Chatbot with FAISS and LLaMA
 
This project is a Streamlit-based chatbot that uses Retrieval-Augmented Generation (RAG) to answer questions about uploaded PDF documents. It leverages FAISS for efficient vector search, HuggingFace sentence transformers for embeddings, and LLaMA (via Ollama) as the language model.
 
## Features
 
- **PDF Upload:** Upload any PDF document.
- **Text Extraction:** Extracts text from all pages using PyPDF2.
- **Chunking & Embedding:** Splits text into chunks and generates embeddings with HuggingFace models.
- **FAISS Vector Store:** Stores and retrieves document chunks using FAISS for fast similarity search.
- **RAG QA Chain:** Combines document retrieval with LLaMA-based question answering.
- **Interactive UI:** Ask questions about your PDF and get context-aware answers.


## Architecture 
<img width="1206" height="608" alt="image" src="https://github.com/user-attachments/assets/00ad2091-b077-43b1-ad69-8a5c0046ccfd" />


 
## Project Structure
 
```
app.py                      # Main Streamlit application
requirements.txt            # Python dependencies
faiss_index/                # Stores FAISS index files
uploaded/                   # Uploaded PDF files
.devcontainer/              # Dev container configuration
```
 
## Setup
 
1. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
 
2. **Run the Streamlit app:**
   ```sh
   streamlit run app.py
   ```
 
3. **Upload a PDF:**  
   Use the web interface to upload your PDF and start asking questions.
 
## Requirements
 
- `streamlit` - Web UI framework
- `PyPDF2` - PDF text extraction
- `langchain` & `langchain-community` - RAG pipeline and integrations
- `sentence-transformers` - Embedding model
- `faiss-cpu` - Vector similarity search
- `ollama` - LLaMA language model backend
 
## How It Works
 
1. **Upload PDF:**  
   The app saves your PDF to the `uploaded/` directory.
 
2. **Text Extraction:**  
   All text is extracted from the PDF.
 
3. **Chunking & Embedding:**  
   Text is split into manageable chunks and converted to vector embeddings.
 
4. **FAISS Indexing:**  
   Chunks are indexed for fast similarity search.
 
5. **Question Answering:**  
   When you ask a question, the app retrieves relevant chunks and uses LLaMA to generate an answer.
 
## Customization
 
- **Change Embedding Model:**  
  Edit the model name in `app.py` (`HuggingFaceEmbeddings`).
 
- **Change LLM Model:**  
  Update the model name in the `Ollama` initialization.
 
## Notes
 
- The FAISS index is stored in `faiss_index/`.
- Uploaded PDFs are saved in `uploaded/`.
- Make sure Ollama and the required LLaMA model are available on your system.
 
 
---# Gen_AI
