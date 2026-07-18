import json

from rag.embedding import EmbeddingModel

from rag.vector_store import VectorStore

from rag.bm25_store import BM25Store

from rag.schemas import DocumentChunk


def main():
    print("Loading chunks...")

    with open(
        "metadata/home_loan_chunks.json",
        encoding="utf8"
    ) as f:

        raw = json.load(f)

    chunks = [

        DocumentChunk(**item)

        for item in raw

    ]

    texts = [

        chunk.text

        for chunk in chunks

    ]

    print(f"{len(chunks)} chunks loaded.")

    embedder = EmbeddingModel()

    embeddings = embedder.encode_documents(texts)

    vector_store = VectorStore()

    vector_store.build(embeddings, chunks)

    vector_store.save()

    bm25 = BM25Store()

    bm25.build(chunks)

    bm25.save()

    print("Done.")


if __name__ == "__main__":
    main()
