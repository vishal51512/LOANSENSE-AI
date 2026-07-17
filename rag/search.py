from rag.embedding import EmbeddingModel

from rag.vector_store import VectorStore

from rag.bm25_store import BM25Store


embedder = EmbeddingModel()

vector = VectorStore()

vector.load()

bm25 = BM25Store()

bm25.load()


query = input("Question : ")

query_embedding = embedder.encode_query(
    query
)

print("\nVector Search\n")

vector_results = vector.search(
    query_embedding,
    top_k=5
)

for item in vector_results:

    print("=" * 50)

    print(
        "Score :",
        item["score"]
    )

    print(
        "Page :",
        item["metadata"].page
    )

    print(
        item["metadata"].text[:300]
    )


print("\nBM25 Search\n")

bm_results = bm25.search(
    query,
    top_k=5
)

for doc, score in bm_results:

    print("=" * 50)

    print(
        "Score :",
        score
    )

    print(
        "Page :",
        doc.page
    )

    print(
        doc.text[:300]
    )