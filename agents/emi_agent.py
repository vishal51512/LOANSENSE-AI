class EMIAgent:

    def invoke(self, state):

        state["emi"] = {

            "message":

            "EMI calculator coming tomorrow."

        }

        return state