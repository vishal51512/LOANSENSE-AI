class ResponseFormatter:

    @staticmethod
    def format(answer, chunks):

        sources = []

        for chunk, score in chunks:

            sources.append(

                {
                    "page": chunk.page,

                    "score": round(score,3)
                }

            )

        return {

            "answer": answer,

            "sources": sources
        }