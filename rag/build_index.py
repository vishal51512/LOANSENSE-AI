"""Build a clean search index from every PDF in the project knowledge base."""

from pathlib import Path

from rag.bm25_store import BM25Store
from rag.embedding import EmbeddingModel
from rag.pipeline import RAGPipeline
from rag.vector_store import VectorStore


DOCUMENT_FOLDERS = (Path("data"), Path("uploads"))


def main():
    pdf_paths = [
        path
        for folder in DOCUMENT_FOLDERS
        if folder.is_dir()
        for path in sorted(folder.glob("*.pdf"))
    ]

    if not pdf_paths:
        raise FileNotFoundError("No PDFs found in data/ or uploads/.")

    pipeline = RAGPipeline()
    chunks = []

    for pdf_path in pdf_paths:
        document_chunks = pipeline.process_pdf(
            pdf_path, bank="SBI", loan_type="Home Loan"
        )
        chunks.extend(document_chunks)
        print(f"Processed {pdf_path}: {len(document_chunks)} chunks")

    if not chunks:
        raise ValueError("No text chunks could be extracted from the available PDFs.")

    embedder = EmbeddingModel()
    embeddings = embedder.encode_documents([chunk.text for chunk in chunks])

    vector_store = VectorStore()
    vector_store.build(embeddings, chunks)
    vector_store.save()

    bm25 = BM25Store()
    bm25.build(chunks)
    bm25.save()

    print(f"Built indexes containing {len(chunks)} chunks.")


if __name__ == "__main__":
    main()
