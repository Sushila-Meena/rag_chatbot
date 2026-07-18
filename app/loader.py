from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

from app.config import DOCS_FOLDER


def load_markdown_files(docs_folder: str = DOCS_FOLDER) -> list[Document]:
    """
    reads every .md file in docs_folder.
    injects the filename into metadata so retrieval results stay traceable.
    raises if the folder is missing or contains no markdown files.
    """
    folder = Path(docs_folder)

    if not folder.exists():
        raise FileNotFoundError(f"docs folder not found: {folder}")

    md_files = sorted(folder.glob("*.md"))

    if not md_files:
        raise FileNotFoundError(f"no .md files found in {folder}")

    all_docs: list[Document] = []

    for file_path in md_files:
        loader = TextLoader(str(file_path), encoding="utf-8")
        docs = loader.load()

        for doc in docs:
            # overwrite langchain's full-path source with just the filename
            doc.metadata["source"] = file_path.name

        all_docs.extend(docs)

    print(f"[loader] loaded {len(all_docs)} file(s) from {folder}")
    return all_docs
