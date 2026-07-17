class PromptBuilder:

    @staticmethod
    def build(query, chunks):

        context = ""

        for chunk, score in chunks:

            context += f"""
Page: {chunk.page}

{chunk.text}

--------------------------------

"""

        prompt = f"""
You are an expert SBI Loan Advisor.

Use ONLY the provided context.

If the answer is not available,
reply exactly:

"I could not find this information in the provided documents."

Never make up information.

Question:

{query}

Context:

{context}

Answer:
"""

        return prompt