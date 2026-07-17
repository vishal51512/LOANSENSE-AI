from dataclasses import dataclass


@dataclass
class DocumentPage:
    page: int
    text: str


@dataclass
class DocumentChunk:
    chunk_id: str
    bank: str
    loan_type: str
    document: str
    page: int
    section: str
    text: str