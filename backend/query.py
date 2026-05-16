from langchain_aws import ChatBedrock, BedrockEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage
import chromadb
from dotenv import load_dotenv

load_dotenv()

embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0",
    region_name="us-east-1"
)
chroma_client = chromadb.PersistentClient(path="./chroma_store")
llm = ChatBedrock(
    model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    region_name="us-east-1"
)

def query_document(doc_id: str, question: str, history: list[dict]) -> tuple[str, list[int]]:
    # 1. Embed the user's question
    q_vec = embeddings.embed_query(question)

    # 2. Retrieve top-5 most relevant chunks
    collection = chroma_client.get_collection(name=doc_id)
    results = collection.query(
        query_embeddings=[q_vec],
        n_results=5,
        include=["documents", "metadatas"]
    )
    chunks = results["documents"][0]
    pages = [m["page"] for m in results["metadatas"][0]]

    # 3. Build context string with page references
    context = "\n\n".join(
        f"[Page {p}] {c}" for p, c in zip(pages, chunks)
    )

    # 4. System prompt
    system = (
        "You are a helpful assistant that answers questions about a document. "
        "Answer using ONLY the provided context. "
        "Always cite the page number inline like (p.3) when referencing information. "
        "If the answer is not in the context, say: 'I could not find that information in the document.'\n\n"
        f"DOCUMENT CONTEXT:\n{context}"
    )

    # 5. Build message list with conversation history (last 3 turns)
    messages = [SystemMessage(content=system)]
    for msg in history[-6:]:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(SystemMessage(content=msg["content"]))
    messages.append(HumanMessage(content=question))

    # 6. Call LLM and return answer + source pages
    response = llm.invoke(messages)
    return response.content, sorted(set(pages))