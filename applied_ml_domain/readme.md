# Multi-Paper RAG Engine

A privacy-focused, fully local Retrieval-Augmented Generation (RAG) pipeline designed to query and summarize multiple research papers simultaneously. 

Architecture
This system operates entirely locally without the need for external API keys:
 Backend:FastAPI 
LLM:LLaMA 3.2 (via Ollama)
Embeddings: HuggingFace (`all-MiniLM-L6-v2`)
Vector Store: ChromaDB
Frontend: Custom HTML/CSS/JS interface

How to Run

1. Install Dependencies
Ensure you are in the root directory (`spider_ml_task_1`) and run:
`pip3 install -r applied_ml_domain/chatbot_code/requirements.txt`

2. Start Ollama
Ensure the Ollama application is running on your machine and the LLaMA 3.2 model is pulled:
`ollama run llama3.2`

3. Ingest the Research Papers
Process the PDFs and build the local Chroma vector database:
`python3 applied_ml_domain/chatbot_code/ingest.py`

4. Start the Backend Server
Launch the FastAPI application:
`python3 -m uvicorn applied_ml_domain.chatbot_code.main:app --reload`

5. Launch the UI
Once the server is running on `http://127.0.0.1:8000`, simply double-click `applied_ml_domain/chatbot_code/index.html` to open the interface in your browser and start asking questions.