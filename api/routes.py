from fastapi import APIRouter, HTTPException

from models.schemas import ChatRequest, ChatResponse, Source


from services.interest_service import InterestService

from agents.orchestrator import Orchestrator

bot = Orchestrator()

router = APIRouter()


interest = InterestService()


@router.get("/health")
def health():

    return {
        "status": "running",
        "project": "LoanSense AI"
    }


@router.get("/interest-rate")
def interest_rate():

    return interest.get_home_loan_rate()


@router.post(
    "/chat",
    response_model=ChatResponse
)
def chat(request: ChatRequest):

    try:

        result = bot.invoke(request.question)

        sources = []

        if "retrieved_docs" in result:

            for chunk, score in result["retrieved_docs"]:

                sources.append(
                    Source(
                        page=chunk.page,
                        score=round(float(score), 3)
                    )
                )

        confidence = 0.0

        if sources:
            confidence = sources[0].score

        return ChatResponse(
            answer=result["answer"],
            confidence=confidence,
            sources=sources
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )