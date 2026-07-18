import re


class QueryClassifier:

    def classify(self, question):

        q = question.lower()

        if re.search(r'\bemi\b', q):
            return "emi"

        if re.search(r'\binterest\s*(rate)?\b', q):
            return "interest"

        return "retrieval"