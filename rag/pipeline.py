import json
from dataclasses import asdict
from pathlib import Path

from rag.loader import PDFLoader
from rag.processor import TextProcessor
from rag.chunker import ChunkGenerator


class RAGPipeline:

    def __init__(self):

        self.loader = PDFLoader()

        self.processor = TextProcessor()

        self.chunker = ChunkGenerator()

    def process_pdf(

        self,

        pdf_path,

        bank,

        loan_type
    ):

        pages = self.loader.load_pdf(pdf_path)

        pages = self.processor.process(pages)

        chunks = self.chunker.create_chunks(

            pages,

            bank,

            loan_type,

            Path(pdf_path).name
        )

        return chunks

    def save_chunks(

        self,

        chunks,

        output_path
    ):

        data = [

            asdict(chunk)

            for chunk in chunks

        ]

        with open(

            output_path,

            "w",

            encoding="utf-8"

        ) as f:

            json.dump(

                data,

                f,

                indent=4,

                ensure_ascii=False
            )