from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import uuid
import os
from ingest import ingest_pdf
from query import query_document

app = FastAPI(title="RAG Document Q&A")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Upload endpoint ──────────────────────────────────────────
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    doc_id = str(uuid.uuid4())
    tmp_path = f"/tmp/{doc_id}.pdf"

    try:
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        num_chunks = ingest_pdf(tmp_path, doc_id)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return {
        "doc_id": doc_id,
        "filename": file.filename,
        "chunks": num_chunks,
        "message": f"Successfully indexed {num_chunks} chunks from '{file.filename}'."
    }


# ── Chat endpoint ────────────────────────────────────────────
class ChatRequest(BaseModel):
    doc_id: str
    question: str
    history: list[dict] = []


@app.post("/chat")
async def chat(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        answer, source_pages = query_document(
            req.doc_id,
            req.question,
            req.history
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

    return {
        "answer": answer,
        "source_pages": source_pages
    }


# ── Health check ─────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok"}


# Run with: uvicorn main:app --reload --port 8000