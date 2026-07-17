from services.interest_service import InterestService

service = InterestService()


class InterestAgent:

    def invoke(self, state):

        state["interest_rate"] = (

            service.get_home_loan_rate()

        )

        return state