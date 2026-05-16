import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_aws import BedrockEmbeddings
import chromadb
import os
from dotenv import load_dotenv

load_dotenv()

embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0",
    region_name="us-east-1"
)
chroma_client = chromadb.PersistentClient(path="./chroma_store")


def ingest_pdf(file_path: str, doc_id: str) -> int:
    # 1. Extract text from each page
    doc = fitz.open(file_path)
    pages = [
        {"text": page.get_text(), "page": i + 1}
        for i, page in enumerate(doc)
    ]
    doc.close()

    # 2. Chunk with overlap
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " "]
    )

    chunks, metadatas, ids = [], [], []
    for page in pages:
        if not page["text"].strip():
            continue
        for j, chunk in enumerate(splitter.split_text(page["text"])):
            chunks.append(chunk)
            metadatas.append({"doc_id": doc_id, "page": page["page"]})
            ids.append(f"{doc_id}_p{page['page']}_c{j}")

    if not chunks:
        raise ValueError("No text could be extracted from this PDF.")

    # 3. Embed and store in ChromaDB
    collection = chroma_client.get_or_create_collection(
        name=doc_id,
        metadata={"hnsw:space": "cosine"}
    )
    vecs = embeddings.embed_documents(chunks)
    collection.add(
        embeddings=vecs,
        documents=chunks,
        metadatas=metadatas,
        ids=ids
    )

    return len(chunks)