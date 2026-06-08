import os
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
import uvicorn

app = FastAPI(title="Multi-Paper RAG Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")
print(f"chroma DB path set to {DB_PATH}.")

embedding_model = None
vector_db = None
llm = None


@app.on_event("startup")
async def startup_event():
    global embedding_model, vector_db, llm
    try:
        print("Initializing embedding model and vector DB (this may take a moment)...")
        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        if not os.path.exists(DB_PATH):
            raise FileNotFoundError(f"Chroma DB directory not found at {DB_PATH}. Run ingestion first.")

        vector_db = Chroma(persist_directory=DB_PATH, embedding_function=embedding_model)
        llm = Ollama(model="llama3.2", temperature=0.0)
        print("Initialization complete: embeddings, vector DB, and LLM ready.")
    except Exception as e:
        print(f"Startup initialization failed: {e}")

class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[str]

@app.post("/api/query", response_model=QueryResponse)
async def handle_rag_query(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        retrieved_docs = vector_db.max_marginal_relevance_search(
            request.question, k=6, fetch_k=20
        )
        context_blocks = []
        sources_found = set()
        for i, doc in enumerate(retrieved_docs):
            source = doc.metadata.get("source_document", "Unknown Paper")
            sources_found.add(source)
            context_blocks.append(f"[Source: {source}]:\n{doc.page_content}")
        combined_context = "\n\n".join(context_blocks)

        if not combined_context.strip():
            fallback_prompt = f"""You are an expert assistant. The user asked: {request.question}\n\nAnswer the question clearly and concisely using your general knowledge."""
            response = llm.invoke(fallback_prompt)
            return QueryResponse(answer=response, sources=[])

        prompt_template = f"""You are an expert AI research assistant. Your task is to answer the user's question based on the provided Context Fragments from research papers.

    RULES:
    1. Synthesize an answer from the provided context. Be helpful and clear.
    2. If the context directly addresses the question, provide a detailed answer citing the sources.
    3. If the context is only tangentially related, explain what you found and acknowledge the limitation.
    4. Only say "I do not have sufficient information" if the context is completely unrelated to the question.
    5. Cite the source paper names explicitly.

    --- CONTEXT FRAGMENTS ---
    {combined_context}

    --- USER QUESTION ---
    {request.question}

    --- YOUR ANSWER ---
    """
        response = llm.invoke(prompt_template)
        return QueryResponse(answer=response, sources=list(sources_found))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status")
async def status():
    return {
        "ready": vector_db is not None and llm is not None,
        "db_path": DB_PATH,
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000)
