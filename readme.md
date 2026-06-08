Spider ML Inductions: Task 1

This repository contains my complete submission for the Spider ML Inductions (Task 1). It encompasses a deep learning image classification model built from scratch and a fully local, interactive RAG (Retrieval-Augmented Generation) web application.

---

Repository Structure

```text
spider_ml_task_1/
│
├── base_task/                  # Deep Learning & Computer Vision
│   ├── notebooks/              # Jupyter notebook with the CNN pipeline
│   ├── saved_models/           # Saved PyTorch weights (.pth, .pkl)
│   ├── accuracy_loss_plots.png # Training performance graphs
│   ├── submission.csv          # Final test set predictions
│   └── README.md
│
├── applied_ml_domain/          # Natural Language Processing & RAG
│   └── chatbot_code/           # FastAPI backend, HTML/JS frontend, and ChromaDB
│       ├── uploaded_pdfs/      # Source research papers
│       ├── ingest.py           # Vector database ingestion script
│       ├── main.py             # Server and LLM routing logic
│       ├── index.html          # Custom UI
│       └── README.md
│
├── bonus_task/                 # Advanced Implementation
│   ├── code/                   # Already covered earlier in chatbot code itself
│   ├── results/                # Evaluation logs of the comparison feature
│   └── README.md
│
└── README.md                   # Master documentation 