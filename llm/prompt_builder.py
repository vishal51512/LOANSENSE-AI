class PromptBuilder:

    SYSTEM_PROMPT = (
        "You are an expert SBI Loan Advisor. "
        "Answer ONLY using the provided context. "
        "If the answer is not present in the context, reply exactly: "
        '"I could not find this information in the provided documents." '
        "Do not make up information."
    )

    @staticmethod
    def build(query, chunks):

        context = ""

        for item in chunks:

            chunk = item[0] if isinstance(item, tuple) else item

            context += f"""
Page: {chunk.page}

{chunk.text}

--------------------------------

"""

        prompt = f"""Question:
{query}

Context:
{context}

Answer:
"""

        return PromptBuilder.SYSTEM_PROMPT, prompt
