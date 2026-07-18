from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from app.config import (
    HEADERS_TO_SPLIT_ON,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    RECURSIVE_SEPARATORS,
)


def split(docs: list[Document]) -> list[Document]:
    """
    two-pass splitting strategy:

    pass 1 — MarkdownHeaderTextSplitter
      splits on #, ##, ### boundaries and promotes headers into metadata.
      this keeps semantically related content together (one section per chunk).

    pass 2 — RecursiveCharacterTextSplitter
      any section still larger than chunk_size gets cut further.
      custom separators respect markdown structure before falling back to whitespace.
    """
    # pass 1: split on markdown headers
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=False,  # keep header text inside the chunk content
    )

    header_chunks: list[Document] = []

    for doc in docs:
        splits = md_splitter.split_text(doc.page_content)

        for chunk in splits:
            # carry the original filename forward — header splitter drops it
            chunk.metadata["source"] = doc.metadata.get("source", "")

        header_chunks.extend(splits)

    print(f"[splitter] header chunks: {len(header_chunks)}")

    # pass 2: enforce chunk_size ceiling
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=RECURSIVE_SEPARATORS,
    )

    final_chunks = recursive_splitter.split_documents(header_chunks)

    # drop near-empty chunks that add noise without value
    final_chunks = [c for c in final_chunks if len(c.page_content.strip()) >= 20]

    print(f"[splitter] final chunks : {len(final_chunks)}")
    return final_chunks
