from pathlib import Path

from rag.pipeline import RAGPipeline


def main():

    pdf_path = "data/home_loan.pdf"

    output = "metadata/home_loan_chunks.json"

    Path("metadata").mkdir(exist_ok=True)

    pipeline = RAGPipeline()

    chunks = pipeline.process_pdf(

        pdf_path,

        bank="SBI",

        loan_type="Home Loan"
    )

    pipeline.save_chunks(

        chunks,

        output
    )

    print("=" * 50)

    print("PDF Processed Successfully")

    print(f"Total Chunks : {len(chunks)}")

    print(f"Saved To     : {output}")

    print("=" * 50)


if __name__ == "__main__":
    main()