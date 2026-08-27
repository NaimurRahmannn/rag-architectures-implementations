from uuid import NAMESPACE_URL, uuid5

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from apps.baseline_rag.app.schemas import DocumentInput


class BaselineChunker:
    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")

        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")

        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
        )

    def split(
        self,
        documents: list[DocumentInput],
    ) -> list[Document]:
        chunks: list[Document] = []

        for source_document in documents:
            document = Document(
                page_content=source_document.content,
                metadata={
                    **source_document.metadata,
                    "document_id": source_document.id,
                    "source": source_document.source,
                },
            )

            document_chunks = self._splitter.split_documents([document])

            for chunk_index, chunk in enumerate(document_chunks):
                start_index = int(chunk.metadata.get("start_index", 0))

                chunk_id = str(
                    uuid5(
                        NAMESPACE_URL,
                        ":".join(
                            [
                                source_document.id,
                                str(start_index),
                                chunk.page_content,
                            ]
                        ),
                    )
                )

                chunk.metadata.update(
                    {
                        "chunk_id": chunk_id,
                        "chunk_index": chunk_index,
                    }
                )

                chunks.append(chunk)

        return chunks
