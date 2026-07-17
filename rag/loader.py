from pathlib import Path

import fitz

from rag.schemas import DocumentPage


class PDFLoader:
    """
    Loads a PDF and extracts text page by page.
    """

    def __init__(self):
        pass

    def load_pdf(self, pdf_path: str):

        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(
                f"File not found: {pdf_path}"
            )

        document = fitz.open(pdf_path)

        pages = []

        for page_number, page in enumerate(document, start=1):

            text = page.get_text("text")

            pages.append(
                DocumentPage(
                    page=page_number,
                    text=text
                )
            )

        document.close()

        return pages