from __future__ import annotations

import re
from pathlib import Path

from langchain_core.documents import Document

from apps.hybrid_rag.app.tokenizer import tokenize


FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tests"
    / "fixtures"
    / "authentication_guide.md"
)

SECTION_PATTERN = re.compile(
    r"^## (?P<title>.+?)\n(?P<body>.*?)(?=^## |\Z)",
    re.MULTILINE | re.DOTALL,
)


def load_documents(path: Path = FIXTURE_PATH) -> list[Document]:
    return [
        Document(
            page_content=path.read_text(encoding="utf-8"),
            metadata={
                "source": str(path),
                "document_id": path.stem,
            },
        )
    ]


def chunk_documents(documents: list[Document]) -> list[Document]:
    chunks: list[Document] = []

    for document in documents:
        for match in SECTION_PATTERN.finditer(document.page_content):
            title = match.group("title").strip()

            if title.lower() == "overview":
                continue

            body = match.group("body").strip()
            chunk_index = len(chunks)

            chunks.append(
                Document(
                    page_content=f"{title}\n\n{body}",
                    metadata={
                        **document.metadata,
                        "chunk_id": f"chunk-{chunk_index + 1:03d}",
                        "chunk_index": chunk_index,
                        "section_title": title,
                    },
                )
            )

    return chunks


def print_chunk(chunk: Document) -> None:
    tokens = tokenize(chunk.page_content)

    print("=" * 60)
    print("Chunk ID:", chunk.metadata["chunk_id"])
    print("Section:", chunk.metadata["section_title"])
    print("Token count:", len(tokens))
    print("Tokens:", tokens)
    print()
    print(chunk.page_content)


def main() -> None:
    documents = load_documents()
    chunks = chunk_documents(documents)

    print("Source:", documents[0].metadata["source"])
    print("Chunks:", len(chunks))
    print()

    for chunk in chunks:
        print_chunk(chunk)


if __name__ == "__main__":
    main()
