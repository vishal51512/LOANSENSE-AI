from rag.rag_engine import RAGEngine

from llm.client import LLMClient

from llm.prompt_builder import PromptBuilder

from llm.response import ResponseFormatter

rag = RAGEngine()

llm = LLMClient()

query = input("Question : ")

chunks = rag.search(query)

prompt = PromptBuilder.build(

    query,

    chunks

)

answer = llm.generate(prompt)

response = ResponseFormatter.format(

    answer,

    chunks

)

print(response["answer"])

print()

print("Sources")

for source in response["sources"]:

    print(source)