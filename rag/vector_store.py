import faiss
import numpy as np
import joblib
import os


class VectorStore:

    def __init__(self):

        self.index = None
        self.metadata = []

    def build(self, embeddings, metadata):

        # IndexFlatIP is cosine similarity only when embeddings are unit-normalized.
        assert np.allclose(
            np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-3
        ), "Embeddings must be normalized before indexing."

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dimension)

        self.index.add(embeddings)

        self.metadata = metadata

    def save(self, folder="vector_store"):

        os.makedirs(folder, exist_ok=True)

        faiss.write_index(
            self.index,
            os.path.join(folder, "sbi.index")
        )

        joblib.dump(
            self.metadata,
            os.path.join(folder, "metadata.pkl")
        )

    def load(self, folder="vector_store"):

        self.index = faiss.read_index(
            os.path.join(folder, "sbi.index")
        )

        self.metadata = joblib.load(
            os.path.join(folder, "metadata.pkl")
        )

    def search(self, query_embedding, top_k=10):

        scores, indices = self.index.search(
            np.array([query_embedding], dtype=np.float32),
            top_k
        )

        results = []

        for score, idx in zip(scores[0], indices[0]):

            if idx == -1:
                continue

            results.append({
                "score": float(score),
                "metadata": self.metadata[idx]
            })

        return results

    def add(self, embeddings, metadata):

        if self.index is None:
            raise RuntimeError(
                "Vector index is not loaded. Call build() or load() first."
            )

        self.index.add(embeddings)

        self.metadata.extend(metadata)
