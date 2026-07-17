from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import torch


class Reranker:

    def __init__(self):

        print("Loading Re-ranker...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            "BAAI/bge-reranker-base"
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            "BAAI/bge-reranker-base"
        )

        self.model.eval()

    def rerank(
        self,
        query,
        documents,
        top_k=5
    ):

        pairs = [

            [query, doc.text]

            for doc in documents

        ]

        with torch.no_grad():

            inputs = self.tokenizer(

                pairs,

                padding=True,

                truncation=True,

                return_tensors="pt",

                max_length=512

            )

            scores = self.model(
                **inputs
            ).logits.squeeze()

        scored = list(
            zip(
                documents,
                scores.tolist()
            )
        )

        scored.sort(
            key=lambda x: x[1],
            reverse=True
        )

        return scored[:top_k]