from rank_bm25 import BM25Okapi
import joblib
import os


class BM25Store:

    def __init__(self):

        self.model = None

        self.documents = []

    def build(self,
              chunks):

        self.documents = chunks

        corpus = [

            chunk.text.lower().split()

            for chunk in chunks

        ]

        self.model = BM25Okapi(
            corpus
        )

    def search(self,
               query,
               top_k=10):

        tokenized = query.lower().split()

        scores = self.model.get_scores(
            tokenized
        )

        ranked = sorted(

            zip(
                self.documents,
                scores
            ),

            key=lambda x: x[1],

            reverse=True

        )

        return ranked[:top_k]

    def save(self,
             folder="vector_store"):

        os.makedirs(folder,
                    exist_ok=True)

        joblib.dump(

            self.model,

            f"{folder}/bm25.pkl"

        )

        joblib.dump(

            self.documents,

            f"{folder}/bm25_docs.pkl"

        )

    def load(self,
             folder="vector_store"):

        self.model = joblib.load(

            f"{folder}/bm25.pkl"

        )

        self.documents = joblib.load(

            f"{folder}/bm25_docs.pkl"

        )