class HybridRetriever:

    def __init__(
        self,
        vector_store,
        bm25_store,
        embedder
    ):

        self.vector = vector_store

        self.bm25 = bm25_store

        self.embedder = embedder

    def search(
        self,
        query,
        top_k=20
    ):

        embedding = self.embedder.encode_query(
            query
        )

        vector_results = self.vector.search(
            embedding,
            top_k
        )

        bm25_results = self.bm25.search(
            query,
            top_k
        )

        merged = {}

        for item in vector_results:

            chunk = item["metadata"]

            merged[
                chunk.chunk_id
            ] = chunk

        for chunk, _ in bm25_results:

            merged[
                chunk.chunk_id
            ] = chunk

        return list(
            merged.values()
        )