import logging

from rag.rag_engine import RAGEngine

logger = logging.getLogger(__name__)


class RetrievalAgent:

    def __init__(self):
        self.rag = RAGEngine()

    def invoke(self, state):

        docs = self.rag.search(state["question"])

        logger.info("Retrieved %d documents", len(docs))

        for i, (doc, score) in enumerate(docs):
            logger.debug("Document %d: %s", i + 1, doc.text[:300])

        state["retrieved_docs"] = docs

        return state
