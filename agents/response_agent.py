import json

from llm.client import LLMClient
from llm.prompt_builder import PromptBuilder


class ResponseAgent:

    def __init__(self):
        self.llm = LLMClient()

    def invoke(self, state):

        system_prompt = None

        if "retrieved_docs" in state:

            system_prompt, prompt = PromptBuilder.build(
                state["question"],
                state["retrieved_docs"]
            )

        elif "interest_rate" in state:

            rate_info = json.dumps(state["interest_rate"], indent=2)

            prompt = f"""Current Interest Data:

{rate_info}

Answer the user's question using only this information.

Question:
{state['question']}
"""

        elif "emi" in state:

            emi_info = json.dumps(state["emi"], indent=2)

            prompt = f"""EMI Information:

{emi_info}

Question:
{state['question']}
"""

        else:

            prompt = f"""Question:
{state['question']}

No additional context is available.
"""

        state["answer"] = self.llm.generate(
            prompt,
            system_prompt=system_prompt
        )

        return state
