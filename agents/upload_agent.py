import os
import logging

from rag.loader import PDFLoader
from rag.processor import TextProcessor
from rag.chunker import ChunkGenerator
from rag.embedding import EmbeddingModel
from rag.vector_store import VectorStore
from rag.bm25_store import BM25Store

logger = logging.getLogger(__name__)


class UploadAgent:

    def __init__(self):

        self.loader = PDFLoader()
        self.processor = TextProcessor()
        self.chunker = ChunkGenerator()
        self.embedder = EmbeddingModel()
        self.store = VectorStore()
        self.bm25 = BM25Store()

    def ingest(self, pdf_path, bank="SBI", loan_type="Home Loan"):

        pages = self.loader.load_pdf(pdf_path)

        pages = self.processor.process(pages)

        chunks = self.chunker.create_chunks(
            pages,
            bank,
            loan_type,
            os.path.basename(pdf_path)
        )

        embeddings = self.embedder.encode_documents(
            [chunk.text for chunk in chunks]
        )

        index_path = os.path.join(
            "vector_store",
            "sbi.index"
        )

        if os.path.exists(index_path):
            self.store.load()
            self.store.add(embeddings, chunks)
        else:
            self.store.build(embeddings, chunks)

        self.store.save()

        all_chunks = self.store.metadata
        self.bm25.build(all_chunks)
        self.bm25.save()

        logger.info(
            "Ingested %d chunks from %s",
            len(chunks), pdf_path
        )

        return len(chunks)