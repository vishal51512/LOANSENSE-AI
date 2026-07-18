import re

from rag.schemas import DocumentPage


class TextProcessor:

    @staticmethod
    def clean_text(text: str):

        text = re.sub(r"\n{3,}", "\n\n", text)

        text = re.sub(r"[ \t]+", " ", text)

        text = text.strip()

        return text

    def process(self, pages):

        cleaned_pages = []

        for page in pages:

            cleaned = self.clean_text(page.text)

            if cleaned:

                cleaned_pages.append(
                    DocumentPage(
                        page=page.page,
                        text=cleaned
                    )
                )

        return cleaned_pages