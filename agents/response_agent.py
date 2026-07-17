from llm.client import LLMClient

from llm.prompt_builder import PromptBuilder

llm = LLMClient()


class ResponseAgent:

    def invoke(self, state):

        if "retrieved_docs" in state:

            prompt = PromptBuilder.build(
                state["question"],
                state["retrieved_docs"]
            )

        elif "interest_rate" in state:

            prompt = f"""
Current Interest Data

{state['interest_rate']}

Answer the user's question using only this information.

Question:
{state['question']}
"""

        elif "emi" in state:

            prompt = f"""
EMI Information

{state['emi']}

Question:
{state['question']}
"""

        else:

            prompt = f"""
Question:
{state['question']}

No additional context is available.
"""

        state["answer"] = llm.generate(prompt)

        return state