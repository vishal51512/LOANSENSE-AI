class ValidationAgent:

    def invoke(self, state):

        answer = state.get("answer", "")

        if not answer or len(answer) < 20:

            state["answer"] = (
                "Unable to generate answer."
            )

        return state