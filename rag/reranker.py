from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import torch


class Reranker:

    def __init__(self, max_documents=50):

        print("Loading Re-ranker...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            "BAAI/bge-reranker-base"
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            "BAAI/bge-reranker-base"
        )

        self.model.eval()
        self.max_documents = max_documents

    def rerank(self, query, documents, top_k=5):

        if not documents:
            return []

        documents = documents[:self.max_documents]

        pairs = [
            [query, item[0].text]
            for item in documents
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
            ).logits.view(-1)

        scored = sorted(
            ((item[0], float(score)) for item, score in zip(documents, scores.tolist())),
            key=lambda x: x[1],
            reverse=True
        )

        return scored[:top_k]
