from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.schemas import DocumentChunk


class ChunkGenerator:

    def __init__(self):

        self.splitter = RecursiveCharacterTextSplitter(

            chunk_size=700,

            chunk_overlap=150
        )

    def create_chunks(

        self,

        pages,

        bank,

        loan_type,

        document_name
    ):

        chunks = []

        counter = 1

        for page in pages:

            splits = self.splitter.split_text(page.text)

            for split in splits:

                chunk = DocumentChunk(

                    chunk_id=f"{bank}_{counter:05d}",

                    bank=bank,

                    loan_type=loan_type,

                    document=document_name,

                    page=page.page,

                    section="Unknown",

                    text=split
                )

                chunks.append(chunk)

                counter += 1

        return chunks