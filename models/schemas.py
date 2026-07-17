from pydantic import BaseModel
from typing import List


class ChatRequest(BaseModel):
    question: str
    bank: str = "SBI"
    loan_type: str = "Home Loan"


class Source(BaseModel):
    page: int
    score: float


class ChatResponse(BaseModel):
    answer: str
    confidence: float
    sources: List[Source]