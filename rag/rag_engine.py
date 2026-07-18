from rag.embedding import EmbeddingModel
from rag.vector_store import VectorStore
from rag.bm25_store import BM25Store
from rag.hybrid import HybridRetriever
from rag.reranker import Reranker


class RAGEngine:

    def __init__(self):

        self.embedder = EmbeddingModel()
        self.vector = VectorStore()
        self.bm25 = BM25Store()
        self.reranker = Reranker()

        self.vector.load()
        self.bm25.load()
        self.hybrid = HybridRetriever(self.vector, self.bm25, self.embedder)

    def reload(self):
        """Reload indexes from disk after new documents are ingested."""
        self.vector.load()
        self.bm25.load()

    def search(self, query, top_k=5):

        retrieved = self.hybrid.search(query)

        ranked = self.reranker.rerank(
            query,
            retrieved,
            top_k
        )

        return ranked
