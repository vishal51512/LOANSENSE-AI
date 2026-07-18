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

        scores = {}
        chunks = {}
        k = 60  # RRF constant

        for rank, item in enumerate(vector_results):
            chunk = item["metadata"]
            cid = chunk.chunk_id
            chunks[cid] = chunk
            scores[cid] = scores.get(cid, 0) + 1.0 / (k + rank + 1)

        for rank, (chunk, _) in enumerate(bm25_results):
            cid = chunk.chunk_id
            chunks[cid] = chunk
            scores[cid] = scores.get(cid, 0) + 1.0 / (k + rank + 1)

        ranked = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return [
            (chunks[cid], score)
            for cid, score in ranked[:top_k]
        ]
