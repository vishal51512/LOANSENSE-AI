class ValidationAgent:

    def invoke(self, state):

        if len(state["answer"]) < 20:

            state["answer"] = (

                "Unable to generate answer."

            )

        return state