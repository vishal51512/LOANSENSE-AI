from services.interest_service import InterestService


class InterestAgent:

    def __init__(self):
        self.service = InterestService()

    def invoke(self, state):

        state["interest_rate"] = (
            self.service.get_home_loan_rate()
        )

        return state