class QueryClassifier:

    def classify(self, question):

        q = question.lower()

        if "emi" in q:

            return "emi"

        if "interest" in q:

            return "interest"

        return "retrieval"