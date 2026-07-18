from agents.classifier import QueryClassifier
from agents.validation_agent import ValidationAgent
from llm.prompt_builder import PromptBuilder
from llm.response import ResponseFormatter
from rag.hybrid import HybridRetriever
from rag.processor import TextProcessor
from rag.schemas import DocumentChunk


def make_chunk(chunk_id="sbi_test_00001"):
    return DocumentChunk(
        chunk_id=chunk_id,
        bank="SBI",
        loan_type="Home Loan",
        document="test.pdf",
        page=1,
        section="Unknown",
        text="Loan eligibility details.",
    )


def test_response_formatter_accepts_ranked_chunks():
    response = ResponseFormatter.format("An answer", [(make_chunk(), 0.875)])

    assert response == {
        "answer": "An answer",
        "sources": [{"page": 1, "score": 0.875}],
    }


def test_prompt_builder_returns_system_and_user_messages():
    system_prompt, user_prompt = PromptBuilder.build("What is eligibility?", [make_chunk()])

    assert "SBI Loan Advisor" in system_prompt
    assert "What is eligibility?" in user_prompt
    assert "Loan eligibility details." in user_prompt


def test_classifier_uses_word_boundaries():
    classifier = QueryClassifier()

    assert classifier.classify("Calculate my EMI") == "emi"
    assert classifier.classify("What is the interest rate?") == "interest"
    assert classifier.classify("Is this semi-annual payment allowed?") == "retrieval"


def test_processor_preserves_paragraph_breaks():
    assert TextProcessor.clean_text("First\n\n\nSecond\t  value") == "First\n\nSecond value"


def test_validation_handles_missing_answer():
    state = ValidationAgent().invoke({})

    assert state["answer"] == "Unable to generate answer."


def test_hybrid_retriever_returns_rrf_scores():
    first, second = make_chunk("first"), make_chunk("second")

    class Embedder:
        def encode_query(self, query):
            return query

    class VectorStore:
        def search(self, embedding, top_k):
            return [{"metadata": first}, {"metadata": second}]

    class BM25Store:
        def search(self, query, top_k):
            return [(first, 1.0), (second, 0.5)]

    results = HybridRetriever(VectorStore(), BM25Store(), Embedder()).search("test")

    assert [chunk.chunk_id for chunk, _ in results] == ["first", "second"]
    assert results[0][1] > results[1][1]
