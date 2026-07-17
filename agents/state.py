from typing import TypedDict


class LoanState(TypedDict):

    question: str

    intent: str

    retrieved_docs: list

    answer: str

    interest_rate: dict

    emi: dict