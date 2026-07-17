from agents.classifier import QueryClassifier

from agents.retrieval_agent import RetrievalAgent

from agents.interest_agent import InterestAgent

from agents.emi_agent import EMIAgent

from agents.response_agent import ResponseAgent

from agents.validation_agent import ValidationAgent


class Orchestrator:

    def __init__(self):

        self.classifier = QueryClassifier()

        self.retrieval = RetrievalAgent()

        self.interest = InterestAgent()

        self.emi = EMIAgent()

        self.response = ResponseAgent()

        self.validation = ValidationAgent()

    def invoke(self, question):

        state = {

            "question": question

        }

        intent = self.classifier.classify(

            question

        )

        if intent == "retrieval":

            state = self.retrieval.invoke(

                state

            )

        elif intent == "interest":

            state = self.interest.invoke(

                state

            )

        else:

            state = self.emi.invoke(

                state

            )

        state = self.response.invoke(

            state
        )

        state = self.validation.invoke(

            state
        )

        return state