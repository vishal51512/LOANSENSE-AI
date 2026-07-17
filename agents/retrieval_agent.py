from rag.rag_engine import RAGEngine

rag = RAGEngine()


class RetrievalAgent:

    def invoke(self, state):

        docs = rag.search(

            state["question"]

        )

        state["retrieved_docs"] = docs

        return state