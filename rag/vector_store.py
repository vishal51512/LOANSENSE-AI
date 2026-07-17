import faiss
import numpy as np
import joblib
import os


class VectorStore:

    def __init__(self):

        self.index = None

        self.metadata = []

    def build(self,
              embeddings,
              metadata):

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(
            dimension
        )

        self.index.add(embeddings)

        self.metadata = metadata

    def save(self,
             folder="vector_store"):

        os.makedirs(folder,
                    exist_ok=True)

        faiss.write_index(
            self.index,
            f"{folder}/sbi.index"
        )

        joblib.dump(
            self.metadata,
            f"{folder}/metadata.pkl"
        )

    def load(self,
             folder="vector_store"):

        self.index = faiss.read_index(
            f"{folder}/sbi.index"
        )

        self.metadata = joblib.load(
            f"{folder}/metadata.pkl"
        )

    def search(self,
               query_embedding,
               top_k=10):

        scores, indices = self.index.search(
            np.array([query_embedding]),
            top_k
        )

        results = []

        for score, idx in zip(
            scores[0],
            indices[0]
        ):

            results.append({

                "score": float(score),

                "metadata":
                self.metadata[idx]

            })

        return results