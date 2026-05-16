# AI Document Q&A System — RAG Pipeline with Amazon Bedrock

A full-stack AI application that lets you upload any PDF and have a multi-turn conversation with it. Built on a production-grade RAG (Retrieval-Augmented Generation) pipeline using Amazon Bedrock, LangChain, and ChromaDB.

![Python](https://img.shields.io/badge/Python-3.13-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)
![LangChain](https://img.shields.io/badge/LangChain-latest-orange)
![AWS Bedrock](https://img.shields.io/badge/AWS-Bedrock-yellow)
![React](https://img.shields.io/badge/React-18-61DAFB)

---

## How It Works

```
PDF Upload → Text Extraction → Chunking → Embedding → ChromaDB
                                                            ↓
User Question → Query Embedding → Similarity Search → Top-K Chunks
                                                            ↓
                                            Claude Sonnet 4.5 → Answer + Page Citations
```

1. **Ingestion** — PDF is extracted page by page using PyMuPDF, split into 512-token chunks with 50-token overlap, embedded using Amazon Titan Embeddings V2, and stored in ChromaDB
2. **Retrieval** — User query is embedded with the same model, and the top-5 most semantically similar chunks are retrieved via cosine similarity search
3. **Synthesis** — Retrieved chunks are injected into a prompt sent to Claude Sonnet 4.5 on Amazon Bedrock, which generates a grounded answer with inline page citations
4. **Memory** — Conversation history is maintained across turns, enabling natural follow-up questions

---

## Tech Stack

| Layer          | Technology                           |
| -------------- | ------------------------------------ |
| LLM            | Amazon Bedrock — Claude Sonnet 4.5   |
| Embeddings     | Amazon Bedrock — Titan Embeddings V2 |
| Vector Store   | ChromaDB (persisted to disk)         |
| RAG Framework  | LangChain, LangChain-AWS             |
| Backend        | FastAPI, Uvicorn                     |
| PDF Processing | PyMuPDF (fitz)                       |
| Frontend       | React, Axios                         |

---

## Project Structure

```
rag-doc-qa/
├── backend/
│   ├── main.py          # FastAPI app — /upload and /chat endpoints
│   ├── ingest.py        # PDF extraction, chunking, embedding, vector storage
│   ├── query.py         # Semantic retrieval and LLM synthesis
│   ├── requirements.txt
│   └── chroma_store/    # Persisted vector database (auto-created)
└── frontend/
    └── src/
        ├── App.jsx
        ├── Upload.jsx
        └── Chat.jsx
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+
- AWS account with Bedrock access enabled for:
  - `amazon.titan-embed-text-v2:0`
  - `us.anthropic.claude-sonnet-4-5-20250929-v1:0`

### Backend Setup

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the `backend/` folder:

```
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=us-east-1
```

Start the server:

```bash
uvicorn main:app --reload --port 8000
```

API will be live at `http://localhost:8000`
Interactive docs at `http://localhost:8000/docs`

### Frontend Setup

```bash
cd frontend
npm install
npm start
```

Frontend will be live at `http://localhost:3000`

---

## API Endpoints

### `POST /upload`

Upload a PDF to index.

**Request:** `multipart/form-data` with a `file` field

**Response:**

```json
{
  "doc_id": "uuid",
  "filename": "document.pdf",
  "chunks": 42,
  "message": "Successfully indexed 42 chunks from 'document.pdf'."
}
```

### `POST /chat`

Ask a question about an indexed document.

**Request:**

```json
{
  "doc_id": "uuid-from-upload",
  "question": "What are the key findings?",
  "history": []
}
```

**Response:**

```json
{
  "answer": "The key findings include... (p.3) ... (p.7)",
  "source_pages": [3, 7]
}
```

### `GET /health`

Returns `{"status": "ok"}` — used for deployment health checks.

---

## Key Features

- **Semantic search** — finds relevant content by meaning, not just keywords
- **Page citations** — every answer references the exact pages it drew from
- **Multi-turn memory** — follow-up questions maintain conversation context
- **Chunking strategy** — 512-token chunks with 50-token overlap preserves context across boundaries
- **Cosine similarity** — ChromaDB collection configured with HNSW cosine distance for accurate retrieval

---

## Environment Variables

| Variable                | Description                     |
| ----------------------- | ------------------------------- |
| `AWS_ACCESS_KEY_ID`     | AWS IAM access key              |
| `AWS_SECRET_ACCESS_KEY` | AWS IAM secret key              |
| `AWS_DEFAULT_REGION`    | AWS region (default: us-east-1) |

---

## Author

**Mohammad Zaid**

- GitHub: [@m-zaid-mac](https://github.com/m-zaid-mac)
- LinkedIn: [mohammad-zaid](https://www.linkedin.com/in/mohammad-zaid-6a360b276/)
- Email: zaid.m@northeastern.edu
