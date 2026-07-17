from sentence_transformers import SentenceTransformer
import numpy as np


class EmbeddingModel:
    """
    Generates embeddings for documents and queries.
    """

    def __init__(self,
                 model_name="BAAI/bge-small-en-v1.5"):

        print(f"Loading embedding model: {model_name}")

        self.model = SentenceTransformer(
            model_name
        )

    def encode_documents(self, texts):

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True
        )

        return np.array(
            embeddings,
            dtype=np.float32
        )

    def encode_query(self, query):

        embedding = self.model.encode(
            query,
            normalize_embeddings=True
        )

        return np.array(
            embedding,
            dtype=np.float32
        )