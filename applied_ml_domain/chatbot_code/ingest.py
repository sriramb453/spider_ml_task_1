import argparse
import os
import shutil
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PDF_DIR = BASE_DIR / "uploaded_pdfs"
DEFAULT_DB_DIR = BASE_DIR / "chroma_db"


def collect_pdfs(pdf_dir: Path):
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {pdf_dir}. Please add your research papers there."
        )
    return pdf_files


def load_documents(pdf_files):
    docs = []
    for pdf_path in pdf_files:
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        for page in pages:
            page.metadata["source_document"] = pdf_path.name
        docs.extend(pages)
        print(f"Loaded {len(pages)} pages from {pdf_path.name}")
    return docs


def build_vector_store(docs, db_dir: Path):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=120)
    chunks = text_splitter.split_documents(docs)
    print(f"Split documents into {len(chunks)} chunks.")

    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    print("Creating Chroma vector store...")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=str(db_dir),
    )
    vector_db.persist()
    return vector_db


def ingest_pdfs(pdf_dir=None, db_dir=None, recreate=False):
    pdf_dir = Path(pdf_dir) if pdf_dir else DEFAULT_PDF_DIR
    db_dir = Path(db_dir) if db_dir else DEFAULT_DB_DIR

    if recreate and db_dir.exists():
        print(f"Removing existing DB at {db_dir}...")
        shutil.rmtree(db_dir)

    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF directory does not exist: {pdf_dir}")

    pdf_files = collect_pdfs(pdf_dir)
    docs = load_documents(pdf_files)
    if not docs:
        raise ValueError("No documents were loaded from the provided PDFs.")

    db_dir.mkdir(parents=True, exist_ok=True)
    return build_vector_store(docs, db_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Ingest PDF research papers into a Chroma vector store for RAG."
    )
    parser.add_argument(
        "--pdf-dir",
        default=str(DEFAULT_PDF_DIR),
        help="Directory containing PDF research papers.",
    )
    parser.add_argument(
        "--db-dir",
        default=str(DEFAULT_DB_DIR),
        help="Directory where Chroma DB will be persisted.",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Recreate the vector database from scratch.",
    )
    args = parser.parse_args()

    ingest_pdfs(pdf_dir=args.pdf_dir, db_dir=args.db_dir, recreate=args.recreate)



if __name__ == "__main__":
    main()
