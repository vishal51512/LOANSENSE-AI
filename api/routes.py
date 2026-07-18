import os
import shutil
import logging

from fastapi import APIRouter, HTTPException, UploadFile, File

from models.schemas import ChatRequest, ChatResponse, Source
from services.interest_service import InterestService
from agents.orchestrator import Orchestrator
from agents.upload_agent import UploadAgent

logger = logging.getLogger(__name__)

router = APIRouter()

bot = Orchestrator()
interest = InterestService()
uploader = UploadAgent()

UPLOAD_FOLDER = "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@router.get("/health")
def health():

    return {
        "status": "running",
        "project": "LoanSense AI"
    }


@router.get("/interest-rate")
def interest_rate():

    return interest.get_home_loan_rate()


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):

    try:

        result = bot.invoke(
            request.question,
            bank=request.bank,
            loan_type=request.loan_type
        )

        docs = result.get("retrieved_docs", [])

        sources = [
            Source(
                page=doc.page,
                score=round(float(score), 3)
            )
            for doc, score in docs
        ]

        return ChatResponse(
            answer=result["answer"],
            confidence=round(sources[0].score, 3) if sources else 0.0,
            sources=sources
        )

    except Exception as e:
        logger.exception("Chat endpoint error")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are accepted."
        )

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        chunks_added = uploader.ingest(file_path)

        # Reload the retrieval agent's RAG engine
        bot.retrieval.rag.reload()

    except Exception as e:
        logger.exception("Upload processing error")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    return {
        "message": f"Successfully processed {file.filename}",
        "chunks_added": chunks_added
    }


@router.get("/knowledge-base")
def knowledge_base():

    uploader.store.load()

    return {
        "bank": "SBI",
        "chunks": len(uploader.store.metadata),
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "reranker": "BAAI/bge-reranker-base"
    }


@router.get("/stats")
def stats():

    uploader.store.load()

    pdf_count = len([
        f for f in os.listdir(UPLOAD_FOLDER)
        if f.lower().endswith(".pdf")
    ]) + len([
        f for f in os.listdir("data")
        if f.lower().endswith(".pdf")
    ])

    return {
        "documents": pdf_count,
        "chunks": len(uploader.store.metadata),
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "reranker": "BAAI/bge-reranker-base",
        "llm": "Llama 3.3 70B",
        "bank": "SBI"
    }


@router.get("/documents")
def documents():

    files = sorted({
        file
        for folder in (UPLOAD_FOLDER, "data")
        if os.path.isdir(folder)
        for file in os.listdir(folder)
        if file.lower().endswith(".pdf")
    })

    return {
        "count": len(files),
        "documents": files
    }
